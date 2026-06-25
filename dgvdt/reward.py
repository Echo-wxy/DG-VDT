"""
DG-VDT — reward engine.

Pure functions over (model output text, observed graph G_obs, training step)
that produce the scalar training signal.

Reward decomposition:
    R_final = R_format + R_graph
    R_graph = sigma(t) * r_strict + (1 - sigma(t)) * r_general + R_collab + R_param
    sigma(t) = 1 / (1 + exp(-k (t - m)))
    r_general = -0.5 + cos(Phi(G_obs), Phi(G*))         (Stage 1, soft)
    r_strict  = 1[ G* subgraph-isomorphic into G_obs ]   (Stage 2, VF2 hard)
    R_collab  = +0.3 if corroboration across >= 2 graph views
    R_param   = -0.3 per misaligned numeric field

Implementation notes:
  * r_general range: cos in [-1, 1] gives a raw r_general in [-1.5, 0.5];
    set `clip_general=True` to clamp to [-0.5, 0.5].
  * VF2 semantics: r_strict tests whether the signature G* is found inside
    G_obs as a (non-induced) subgraph monomorphism -- the classical VF2
    "find pattern in host" use, implemented as
    MultiDiGraphMatcher(host=G_obs, pattern=G*).subgraph_is_monomorphic().
    Set `induced_vf2=True` for node-induced subgraph isomorphism instead.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
from networkx.algorithms import isomorphism as iso

from .encoder import RGCNEncoder
from .schema import TraceGraph, VULN_TYPES

# --- output schema regexes -------------------------------------------------
_VULN_RE = re.compile(r"<vulnerability>\s*(.*?)\s*</vulnerability>", re.IGNORECASE | re.DOTALL)
_TRACE_RE = re.compile(r"<trace>\s*(.*?)\s*</trace>", re.IGNORECASE | re.DOTALL)
_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


@dataclass
class ParsedOutput:
    format_ok: int            # R_format in {0,1}
    vuln_type: str | None     # parsed & validated type, else None
    trace_addr: str | None    # parsed & validated 0x-address, else None


def parse_output(text: str) -> ParsedOutput:
    """R_format gate: well-formed <vulnerability>TYPE</vulnerability> with TYPE in
    VULN_TYPES AND well-formed <trace>0x..40hex..</trace>."""
    vt = _VULN_RE.search(text or "")
    tr = _TRACE_RE.search(text or "")
    vuln = vt.group(1).strip().lower() if vt else None
    addr = tr.group(1).strip() if tr else None
    vuln_ok = vuln in VULN_TYPES
    addr_ok = bool(addr and _ADDR_RE.match(addr))
    return ParsedOutput(
        format_ok=int(vuln_ok and addr_ok),
        vuln_type=vuln if vuln_ok else None,
        trace_addr=addr if addr_ok else None,
    )


# --- Stage 1: cosine similarity reward ------------------------------------
def cosine_reward(
    encoder: RGCNEncoder,
    g_obs: TraceGraph,
    g_star: TraceGraph,
    clip_general: bool = False,
) -> float:
    z_obs = encoder.embed(g_obs.to_pyg())
    z_star = encoder.embed(g_star.to_pyg())
    cos = torch.nn.functional.cosine_similarity(z_obs, z_star, dim=0).item()
    r = -0.5 + cos
    if clip_general:
        r = max(-0.5, min(0.5, r))
    return r


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


# --- curriculum & auxiliary terms -----------------------------------------
def sigma(t: int, k: float, m: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (t - m)))


def collaboration_bonus(g_obs: TraceGraph, g_star: TraceGraph, value: float = 0.3) -> float:
    """+value if the observed graph shows corroborating evidence across >= 2 of
    the graph views the signature relies on (e.g. cyclic FFG + circular CaG).
    Faithful proxy: |views(G_obs) ∩ views(G*)| >= 2."""
    shared = g_obs.views_present() & g_star.views_present()
    return value if len(shared) >= 2 else 0.0


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
    k: float = 0.01          # sigmoid steepness
    m: float = 1000.0        # sigmoid midpoint step
    clip_general: bool = False
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
    """Compute R_final for one sample. The predicted <vulnerability> type selects
    which reference signature G* to score against (this is what grounds the
    textual prediction in graph structure). If the format gate fails, the graph
    reward is withheld (R_format acts as a prerequisite gate)."""
    cfg = cfg or RewardConfig()
    parsed = parse_output(text)

    if not parsed.format_ok or parsed.vuln_type not in references:
        return RewardBreakdown(
            r_format=float(parsed.format_ok),
            r_general=0.0, r_strict=0.0, sigma=sigma(step, cfg.k, cfg.m),
            r_collab=0.0, r_param=0.0, r_graph=0.0,
            r_final=float(parsed.format_ok), vuln_type=parsed.vuln_type,
        )

    g_star = references[parsed.vuln_type]
    s = sigma(step, cfg.k, cfg.m)
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
