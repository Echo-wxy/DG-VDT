"""
DG-VDT — reward engine.

Pure functions over (model output text, observed graph G_obs, training step)
that produce the scalar training signal.

Reward decomposition (paper Eqs. 1-5):
    R_final = R_format + R_graph
    R_graph = sigma(t) * r_strict + (1 - sigma(t)) * r_general + R_collab + R_param
    sigma(t) = 1 / (1 + exp(-k (t - m)))     with t, m in THOUSANDS of steps
    r_general = max(0, cos(Phi(G_obs), Phi(G*))) - 0.5     in [-0.5, +0.5]
    r_strict  = 1[ G* subgraph-isomorphic into G_obs ]     (Stage 2, VF2 hard)
    R_collab  = +0.3 if corroborating ANOMALIES co-occur in >= 2 graph views
    R_param   = -0.3 per misaligned numeric field

Output schema:
    <vulnerability>TYPE</vulnerability> with TYPE in
    {reentrancy, short_address, timestamp, none}; for an attack verdict a
    well-formed <trace>0x..40hex..</trace> is REQUIRED, for a benign verdict
    ("none") the <trace> tag must be OMITTED. A benign verdict receives the
    format reward only: no reference signature G* exists for the benign
    class, so the graph reward is not applicable.

Curriculum convention (paper Sec. "Curriculum Scheduling"):
    t and m are expressed in thousands of training steps and k carries units
    of (10^3 steps)^-1, so the paper's optimum is k=0.3, m=8 (i.e. 8,000 raw
    steps). Callers pass RAW step counts to compute_reward(); the division by
    RewardConfig.step_scale (default 1000) performs the unit conversion.

VF2 semantics: r_strict tests whether the signature G* is found inside
G_obs as a (non-induced) subgraph monomorphism -- the classical VF2
"find pattern in host" use, implemented as
MultiDiGraphMatcher(host=G_obs, pattern=G*).subgraph_is_monomorphic().
Set `induced_vf2=True` for node-induced subgraph isomorphism instead.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import networkx as nx
import torch
from networkx.algorithms import isomorphism as iso

from .encoder import RGCNEncoder
from .schema import BENIGN_TYPE, OUTPUT_TYPES, TraceGraph, VULN_TYPES

# --- output schema regexes -------------------------------------------------
_VULN_RE = re.compile(r"<vulnerability>\s*(.*?)\s*</vulnerability>", re.IGNORECASE | re.DOTALL)
_TRACE_RE = re.compile(r"<trace>\s*(.*?)\s*</trace>", re.IGNORECASE | re.DOTALL)
_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


@dataclass
class ParsedOutput:
    format_ok: int            # R_format in {0,1}
    vuln_type: str | None     # parsed & validated type ("none" for benign), else None
    trace_addr: str | None    # parsed & validated 0x-address, else None


def parse_output(text: str) -> ParsedOutput:
    """R_format gate.

    Attack verdict : <vulnerability>TYPE</vulnerability> with TYPE in
                     VULN_TYPES AND a well-formed <trace>0x..40hex..</trace>.
    Benign verdict : <vulnerability>none</vulnerability> and NO <trace> tag
                     (there is no attacker address to report).
    """
    vt = _VULN_RE.search(text or "")
    tr = _TRACE_RE.search(text or "")
    vuln = vt.group(1).strip().lower() if vt else None
    addr = tr.group(1).strip() if tr else None
    addr_ok = bool(addr and _ADDR_RE.match(addr))

    if vuln == BENIGN_TYPE:
        # benign: the trace tag must be absent
        ok = tr is None
        return ParsedOutput(format_ok=int(ok), vuln_type=BENIGN_TYPE if ok else None,
                            trace_addr=None)

    vuln_ok = vuln in VULN_TYPES
    ok = vuln_ok and addr_ok
    return ParsedOutput(
        format_ok=int(ok),
        vuln_type=vuln if vuln_ok else None,
        trace_addr=addr if addr_ok else None,
    )


# --- Stage 1: cosine similarity reward ------------------------------------
def cosine_reward(
    encoder: RGCNEncoder,
    g_obs: TraceGraph,
    g_star: TraceGraph,
    clip_general: bool = True,
) -> float:
    """Paper Eq. 2: r_general = max(0, cos) - 0.5, guaranteed in [-0.5, +0.5].

    clip_general=False recovers the raw legacy form (-0.5 + cos), whose lower
    bound is -1.5; the paper formula (default) clamps the cosine at 0 first.
    """
    z_obs = encoder.embed(g_obs.to_pyg())
    z_star = encoder.embed(g_star.to_pyg())
    cos = torch.nn.functional.cosine_similarity(z_obs, z_star, dim=0).item()
    if clip_general:
        return max(0.0, cos) - 0.5
    return -0.5 + cos


# --- Stage 2: VF2 subgraph isomorphism reward -----------------------------
def vf2_match(g_obs: TraceGraph, g_star: TraceGraph, induced: bool = False) -> bool:
    """True iff signature g_star is found inside g_obs (node+edge labels matched)."""
    host = g_obs.to_networkx()
    pattern = g_star.to_networkx()
    nm = iso.categorical_node_match("ntype", None)
    em = iso.categorical_multiedge_match("etype", None)
    gm = iso.MultiDiGraphMatcher(host, pattern, node_match=nm, edge_match=em)
    return gm.subgraph_is_isomorphic() if induced else gm.subgraph_is_monomorphic()


def strict_reward(g_obs: TraceGraph, g_star: TraceGraph, induced: bool = False) -> float:
    return 1.0 if vf2_match(g_obs, g_star, induced=induced) else 0.0


# --- curriculum ------------------------------------------------------------
def sigma(t: float, k: float, m: float) -> float:
    """Sigmoid curriculum. t and m in thousands of steps, k in (10^3 steps)^-1."""
    return 1.0 / (1.0 + math.exp(-k * (t - m)))


# --- multi-graph collaboration bonus ---------------------------------------
# Per-view ANOMALY detectors. The bonus is awarded only when corroborating
# anomalies are detected simultaneously in >= 2 graph views (paper Sec.
# "Stage 1"), e.g. a cyclic FFG path co-occurring with a CaG re-entry edge.
# Merely sharing edge views with the signature is NOT sufficient.

def _ffg_anomaly(g: TraceGraph) -> bool:
    """Fund-flow anomaly: a directed Ether cycle (reentrancy drain pattern),
    or value attached in parallel to a truncated-ABI CALL (short-address
    economic corroboration)."""
    eth_edges = [(e.src, e.dst) for e in g.edges if e.etype == "ETH"]
    if eth_edges:
        dg = nx.DiGraph(eth_edges)
        try:
            nx.find_cycle(dg)
            return True
        except nx.NetworkXNoCycle:
            pass
    short_calls = {(e.src, e.dst) for e in g.edges
                   if e.etype == "CALL" and e.attrs.get("arg_len", 32) < 32}
    return any(pair in short_calls for pair in eth_edges)


def _cag_anomaly(g: TraceGraph) -> bool:
    """Call-graph anomaly: (a) re-entry -- a CALL u->v antiparallel to an
    Ether flow v->u; (b) a CALL whose ABI argument length is < 32 bytes
    (short-address); (c) a TIMESTAMP-style gate -- a Branch node fed by an
    Opcode node (timestamp-dependence causal chain)."""
    eth_reversed = {(e.dst, e.src) for e in g.edges if e.etype == "ETH"}
    ntype = {n.nid: n.ntype for n in g.nodes}
    for e in g.edges:
        if e.etype == "CALL":
            if (e.src, e.dst) in eth_reversed:
                return True                    # re-entry
            if e.attrs.get("arg_len", 32) < 32:
                return True                    # truncated ABI argument
        if ntype.get(e.src) == "Opcode" and ntype.get(e.dst) == "Branch":
            return True                        # opcode-gated branch
    return False


def _ccg_anomaly(g: TraceGraph) -> bool:
    """Creation-graph anomaly: a deployment edge originating from a Miner
    account (miner deploy hub), or any deployer with CREATE out-degree >= 2."""
    ntype = {n.nid: n.ntype for n in g.nodes}
    out_deg: dict = {}
    for e in g.edges:
        if e.etype == "CREATE":
            if ntype.get(e.src) == "Miner":
                return True
            out_deg[e.src] = out_deg.get(e.src, 0) + 1
            if out_deg[e.src] >= 2:
                return True
    return False


_VIEW_DETECTORS = {"FFG": _ffg_anomaly, "CaG": _cag_anomaly, "CCG": _ccg_anomaly}


def anomalous_views(g_obs: TraceGraph) -> set[str]:
    """The set of graph views in which a discriminative anomaly is detected."""
    return {view for view, det in _VIEW_DETECTORS.items() if det(g_obs)}


def collaboration_bonus(g_obs: TraceGraph, g_star: TraceGraph | None = None,
                        value: float = 0.3) -> float:
    """+value iff corroborating anomalies co-occur in >= 2 graph views of the
    OBSERVED graph (e.g. FFG fund cycle + CaG re-entry edge; or a CaG
    opcode-gated transfer + a CCG miner deploy hub). g_star is accepted for
    backward compatibility but the decision depends on G_obs only, matching
    the paper's definition of the bonus."""
    return value if len(anomalous_views(g_obs)) >= 2 else 0.0


def param_penalty(pred_params: dict | None, ref_params: dict | None, per_field: float = -0.3) -> float:
    """-0.3 per misaligned numeric field between predicted and reference."""
    if not pred_params or not ref_params:
        return 0.0
    mism = 0
    for key, ref_val in ref_params.items():
        if key in pred_params and pred_params[key] != ref_val:
            mism += 1
    return per_field * mism


# --- full composition ------------------------------------------------------
@dataclass
class RewardConfig:
    k: float = 0.3           # sigmoid steepness, units (10^3 steps)^-1 (paper optimum)
    m: float = 8.0           # sigmoid midpoint in thousands of steps (paper: 8k)
    step_scale: float = 1000.0   # raw steps per curriculum unit (t = step / step_scale)
    clip_general: bool = True    # paper Eq. 2 form: max(0, cos) - 0.5
    induced_vf2: bool = False
    collab_value: float = 0.3
    param_per_field: float = -0.3


@dataclass
class RewardBreakdown:
    r_format: float
    r_general: float
    r_strict: float
    sigma: float
    r_collab: float
    r_param: float
    r_graph: float
    r_final: float
    vuln_type: str | None


def compute_reward(
    text: str,
    g_obs: TraceGraph,
    step: int,
    encoder: RGCNEncoder,
    references: dict[str, TraceGraph],
    cfg: RewardConfig | None = None,
    pred_params: dict | None = None,
    ref_params: dict | None = None,
) -> RewardBreakdown:
    """Compute R_final for one sample.

    `step` is the RAW training step count; the curriculum position is
    t = step / cfg.step_scale (thousands of steps), matching the paper.

    The predicted <vulnerability> type selects which reference signature G*
    to score against (this is what grounds the textual prediction in graph
    structure). A benign verdict ("none") has no G*: it earns the format
    reward only. If the format gate fails, the graph reward is withheld
    (R_format acts as a prerequisite gate).
    """
    cfg = cfg or RewardConfig()
    parsed = parse_output(text)
    s = sigma(step / cfg.step_scale, cfg.k, cfg.m)

    graph_applicable = (
        parsed.format_ok
        and parsed.vuln_type in VULN_TYPES
        and parsed.vuln_type in references
    )
    if not graph_applicable:
        # covers: malformed output, unknown type, and the benign verdict
        return RewardBreakdown(
            r_format=float(parsed.format_ok),
            r_general=0.0, r_strict=0.0, sigma=s,
            r_collab=0.0, r_param=0.0, r_graph=0.0,
            r_final=float(parsed.format_ok), vuln_type=parsed.vuln_type,
        )

    g_star = references[parsed.vuln_type]
    r_general = cosine_reward(encoder, g_obs, g_star, clip_general=cfg.clip_general)
    r_strict = strict_reward(g_obs, g_star, induced=cfg.induced_vf2)
    r_collab = collaboration_bonus(g_obs, g_star, value=cfg.collab_value)
    r_param = param_penalty(pred_params, ref_params, per_field=cfg.param_per_field)

    r_graph = s * r_strict + (1.0 - s) * r_general + r_collab + r_param
    r_final = float(parsed.format_ok) + r_graph
    return RewardBreakdown(
        r_format=float(parsed.format_ok), r_general=r_general, r_strict=r_strict,
        sigma=s, r_collab=r_collab, r_param=r_param, r_graph=r_graph,
        r_final=r_final, vuln_type=parsed.vuln_type,
    )
