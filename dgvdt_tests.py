"""
Toy-graph unit tests for the DG-VDT reward engine.

Run: python -m pytest dgvdt_tests.py -q   (or just `python dgvdt_tests.py`)

The format gate, VF2, curriculum, collaboration bonus and parameter penalty
are deterministic and tested directly. For the cosine term only the
encoder-agnostic property is asserted (identical inputs -> cosine 1); a
meaningful ranking of related vs. unrelated graphs emerges once the encoder
Phi has been contrastively pre-trained and frozen.

Curriculum convention: RewardConfig defaults follow the paper's optimum
(k=0.3, m=8 in thousands of steps); compute_reward takes RAW step counts
and divides by step_scale=1000.
"""
from __future__ import annotations

import math

import torch

from dgvdt import (
    BENIGN_TYPE, Edge, Node, RGCNEncoder, RewardConfig, TraceGraph,
    anomalous_views, canonical_references, collaboration_bonus, compute_reward,
    cosine_reward, param_penalty, parse_output, sigma, strict_reward, vf2_match,
)

REFS = canonical_references()


# --- builders for observed graphs -----------------------------------------
def g_obs_reentrancy_full() -> TraceGraph:
    """A copy of the reentrancy signature plus extra unrelated activity, to
    prove non-induced (monomorphic) matching tolerates a larger host."""
    g = TraceGraph(
        [Node("A", "EOA"), Node("V", "Contract"), Node("Ac", "Contract"),
         Node("b0", "State"), Node("b1", "State"), Node("b2", "State"), Node("b3", "State"),
         Node("X", "EOA"), Node("Y", "Contract")],            # extra noise nodes
        [Edge("A", "V", "ETH"), Edge("V", "Ac", "ETH"), Edge("Ac", "V", "ETH"),
         Edge("Ac", "V", "CALL"), Edge("V", "b0", "SLOAD"), Edge("V", "b1", "SSTORE"),
         Edge("Ac", "b2", "SLOAD"), Edge("Ac", "b3", "SSTORE"),
         Edge("X", "Y", "ETH"), Edge("X", "Y", "CALL")],      # extra noise edges
    )
    return g


def g_obs_reentrancy_broken() -> TraceGraph:
    """Same as full but missing the CaG CALL re-entry edge -> must NOT match."""
    g = g_obs_reentrancy_full()
    g.edges = [e for e in g.edges if not (e.src == "Ac" and e.dst == "V" and e.etype == "CALL")]
    return g


def g_obs_unrelated() -> TraceGraph:
    return TraceGraph(
        [Node("p", "EOA"), Node("q", "EOA")],
        [Edge("p", "q", "ETH")],
    )


def g_obs_benign_transfer_and_call() -> TraceGraph:
    """A perfectly ordinary payment + function call. Shares FFG and CaG views
    with the signatures but contains NO anomaly in any view: the collaboration
    bonus must NOT fire (regression test against the old shared-views proxy)."""
    return TraceGraph(
        [Node("u", "EOA"), Node("c", "Contract"), Node("s", "State")],
        [Edge("u", "c", "ETH"),                       # plain payment      (FFG)
         Edge("u", "c", "CALL", attrs={"arg_len": 68}),  # well-formed call (CaG)
         Edge("c", "s", "SSTORE")],                   # state write        (CaG)
    )


def _valid(vuln: str) -> str:
    return f"<vulnerability>{vuln}</vulnerability> <trace>0x" + "ab" * 20 + "</trace>"


# --- format gate -----------------------------------------------------------
def test_format_valid():
    p = parse_output(_valid("reentrancy"))
    assert p.format_ok == 1 and p.vuln_type == "reentrancy"


def test_format_invalid_type():
    out = "<vulnerability>banana</vulnerability> <trace>0x" + "ab" * 20 + "</trace>"
    assert parse_output(out).format_ok == 0


def test_format_invalid_addr():
    out = "<vulnerability>reentrancy</vulnerability> <trace>0x1234</trace>"
    assert parse_output(out).format_ok == 0


def test_format_missing_tags():
    assert parse_output("I think it is reentrancy at 0xabc").format_ok == 0


def test_format_benign_none_without_trace_ok():
    p = parse_output("<vulnerability>none</vulnerability>")
    assert p.format_ok == 1 and p.vuln_type == BENIGN_TYPE and p.trace_addr is None


def test_format_benign_none_with_trace_rejected():
    # a benign verdict must OMIT the trace tag
    assert parse_output(_valid("none")).format_ok == 0


def test_format_attack_requires_trace():
    assert parse_output("<vulnerability>reentrancy</vulnerability>").format_ok == 0


# --- Stage 2: VF2 ----------------------------------------------------------
def test_vf2_matches_full_signature():
    assert vf2_match(g_obs_reentrancy_full(), REFS["reentrancy"]) is True


def test_vf2_rejects_broken_signature():
    assert vf2_match(g_obs_reentrancy_broken(), REFS["reentrancy"]) is False


def test_vf2_rejects_unrelated():
    assert vf2_match(g_obs_unrelated(), REFS["reentrancy"]) is False


def test_vf2_cross_type_rejection():
    # a reentrancy host should not satisfy the short-address signature
    assert vf2_match(g_obs_reentrancy_full(), REFS["short_address"]) is False


def test_vf2_self_match_all_types():
    # each signature trivially embeds in itself
    for t, g in REFS.items():
        assert vf2_match(g, g) is True, t


def test_strict_reward_binary():
    assert strict_reward(g_obs_reentrancy_full(), REFS["reentrancy"]) == 1.0
    assert strict_reward(g_obs_unrelated(), REFS["reentrancy"]) == 0.0


# --- Stage 1: cosine reward ------------------------------------------------
def test_cosine_identical_graphs_is_half():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    g = REFS["reentrancy"]
    r = cosine_reward(enc, g, g)
    assert abs(r - 0.5) < 1e-5          # cos(z, z) = 1 -> max(0,1)-0.5 = +0.5


def test_cosine_paper_bounds():
    # paper Eq. 2 guarantees r_general in [-0.5, +0.5] for ANY pair
    torch.manual_seed(0)
    enc = RGCNEncoder()
    pairs = [(g_obs_reentrancy_full(), REFS["timestamp"]),
             (g_obs_unrelated(), REFS["short_address"]),
             (g_obs_benign_transfer_and_call(), REFS["reentrancy"])]
    for a, b in pairs:
        r = cosine_reward(enc, a, b)
        assert -0.5 - 1e-9 <= r <= 0.5 + 1e-9


# --- curriculum ------------------------------------------------------------
def test_sigma_paper_defaults():
    cfg = RewardConfig()
    assert cfg.k == 0.3 and cfg.m == 8.0 and cfg.step_scale == 1000.0
    # midpoint: raw step 8000 -> t=8 -> sigma = 0.5
    assert abs(sigma(8000 / cfg.step_scale, cfg.k, cfg.m) - 0.5) < 1e-9
    # start of training: sigma(0) = 1/(1+e^{2.4}) ~ 0.083, Stage 1 dominated
    assert sigma(0.0, cfg.k, cfg.m) < 0.1
    # late training (20k raw steps): Stage 2 dominated
    assert sigma(20000 / cfg.step_scale, cfg.k, cfg.m) > 0.95


def test_sigma_monotone():
    cfg = RewardConfig()
    vals = [sigma(t, cfg.k, cfg.m) for t in (0.0, 4.0, 8.0, 12.0, 20.0)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


# --- collaboration bonus (anomaly co-occurrence, not shared views) ---------
def test_collab_fires_on_reentrancy_co_anomaly():
    # FFG fund cycle + CaG re-entry edge -> two anomalous views -> +0.3
    g = g_obs_reentrancy_full()
    assert {"FFG", "CaG"} <= anomalous_views(g)
    assert collaboration_bonus(g) == 0.3


def test_collab_does_not_fire_without_reentry():
    # fund cycle alone (one anomalous view) is not corroboration
    g = g_obs_reentrancy_broken()
    assert anomalous_views(g) == {"FFG"}
    assert collaboration_bonus(g) == 0.0


def test_collab_does_not_fire_on_benign_shared_views():
    # regression: sharing FFG+CaG views with the signature must NOT pay
    g = g_obs_benign_transfer_and_call()
    assert anomalous_views(g) == set()
    assert collaboration_bonus(g) == 0.0


def test_collab_fires_on_timestamp_cag_ccg():
    # TD signature: opcode-gated branch (CaG) + miner deploy hub (CCG)
    g = REFS["timestamp"]
    assert {"CaG", "CCG"} <= anomalous_views(g)
    assert collaboration_bonus(g) == 0.3


def test_collab_fires_on_short_address():
    # SA signature: truncated-ABI CALL (CaG) + value attached in parallel (FFG)
    g = REFS["short_address"]
    assert {"FFG", "CaG"} <= anomalous_views(g)
    assert collaboration_bonus(g) == 0.3


# --- parameter penalty ------------------------------------------------------
def test_param_penalty_counts_mismatches():
    assert param_penalty({"amount": 1.0, "order": 2}, {"amount": 1.0, "order": 3}) == -0.3
    assert param_penalty({"amount": 9.9}, {"amount": 1.0}) == -0.3
    assert param_penalty({"amount": 1.0}, {"amount": 1.0}) == 0.0
    assert param_penalty(None, {"amount": 1.0}) == 0.0


# --- full composition -------------------------------------------------------
def test_full_late_step_true_positive():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    rb = compute_reward(_valid("reentrancy"), g_obs_reentrancy_full(), step=20000,
                        encoder=enc, references=REFS)
    # late step: sigma~1 -> r_graph ~ r_strict(1) + collab(0.3); + r_format(1)
    assert rb.sigma > 0.95
    assert rb.r_strict == 1.0
    assert rb.r_collab == 0.3
    assert rb.r_final > 2.0


def test_full_late_step_true_negative():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    rb = compute_reward(_valid("reentrancy"), g_obs_reentrancy_broken(), step=20000,
                        encoder=enc, references=REFS)
    assert rb.r_strict == 0.0
    assert rb.r_collab == 0.0        # single-view anomaly: no corroboration
    assert rb.r_final < 2.0


def test_full_benign_verdict_gets_format_reward_only():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    rb = compute_reward("<vulnerability>none</vulnerability>",
                        g_obs_benign_transfer_and_call(), step=20000,
                        encoder=enc, references=REFS)
    assert rb.r_format == 1.0 and rb.vuln_type == BENIGN_TYPE
    assert rb.r_graph == 0.0 and rb.r_final == 1.0


def test_full_format_gate_blocks_graph_reward():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    rb = compute_reward("no tags here", g_obs_reentrancy_full(), step=20000,
                        encoder=enc, references=REFS)
    assert rb.r_format == 0.0 and rb.r_graph == 0.0 and rb.r_final == 0.0


def test_full_early_step_is_general_dominated():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    rb = compute_reward(_valid("reentrancy"), g_obs_reentrancy_full(), step=0,
                        encoder=enc, references=REFS)
    assert rb.sigma < 0.1            # strict barely weighted this early


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
