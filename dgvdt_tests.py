"""
Toy-graph unit tests for the DG-VDT reward engine.

Run: python -m pytest dgvdt_tests.py -q   (or just `python dgvdt_tests.py`)

The format gate, VF2, curriculum, collaboration bonus and parameter penalty
are deterministic and tested directly. For the cosine term only the
encoder-agnostic property is asserted (identical inputs -> cosine 1); a
meaningful ranking of related vs. unrelated graphs emerges once the encoder
Phi has been contrastively pre-trained and frozen.
"""
from __future__ import annotations

import math

import torch

from dgvdt import (
    Edge, Node, RGCNEncoder, RewardConfig, TraceGraph, canonical_references,
    collaboration_bonus, compute_reward, cosine_reward, param_penalty,
    parse_output, sigma, strict_reward, vf2_match,
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


# --- format gate -----------------------------------------------------------
def test_format_valid():
    out = "<vulnerability>reentrancy</vulnerability> <trace>0x" + "ab" * 20 + "</trace>"
    p = parse_output(out)
    assert p.format_ok == 1 and p.vuln_type == "reentrancy"


def test_format_invalid_type():
    out = "<vulnerability>banana</vulnerability> <trace>0x" + "ab" * 20 + "</trace>"
    assert parse_output(out).format_ok == 0


def test_format_invalid_addr():
    out = "<vulnerability>reentrancy</vulnerability> <trace>0x1234</trace>"
    assert parse_output(out).format_ok == 0


def test_format_missing_tags():
    assert parse_output("I think it is reentrancy at 0xabc").format_ok == 0


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


# --- Stage 1: cosine (encoder-agnostic properties only) --------------------
def test_cosine_identity_is_half():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    # cos(G*, G*) == 1  ->  r_general = -0.5 + 1 = 0.5  (the stated maximum)
    r = cosine_reward(enc, REFS["reentrancy"], REFS["reentrancy"])
    assert abs(r - 0.5) < 1e-5


def test_cosine_in_raw_range():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    r = cosine_reward(enc, g_obs_unrelated(), REFS["reentrancy"])
    assert -1.5 - 1e-6 <= r <= 0.5 + 1e-6


def test_cosine_clip_option():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    r = cosine_reward(enc, g_obs_unrelated(), REFS["reentrancy"], clip_general=True)
    assert -0.5 - 1e-6 <= r <= 0.5 + 1e-6


# --- curriculum sigmoid ----------------------------------------------------
def test_sigmoid_endpoints_and_midpoint():
    k, m = 0.01, 1000.0
    assert sigma(m, k, m) == 0.5
    assert sigma(0, k, m) < 0.01           # early: general-dominated
    assert sigma(5000, k, m) > 0.99        # late: strict-dominated


# --- collaboration bonus ---------------------------------------------------
def test_collab_awarded_multiview():
    # reentrancy G* spans FFG+CaG; full host shares both -> +0.3
    assert collaboration_bonus(g_obs_reentrancy_full(), REFS["reentrancy"]) == 0.3


def test_collab_withheld_single_view():
    # unrelated host has only FFG -> shares <2 views -> 0
    assert collaboration_bonus(g_obs_unrelated(), REFS["reentrancy"]) == 0.0


# --- param penalty ---------------------------------------------------------
def test_param_penalty_counts_mismatches():
    pred = {"amount": 100, "addr": "0xA"}
    ref = {"amount": 100, "addr": "0xB"}      # 1 mismatch
    assert abs(param_penalty(pred, ref) - (-0.3)) < 1e-9


def test_param_penalty_zero_when_absent():
    assert param_penalty(None, None) == 0.0


# --- full composition ------------------------------------------------------
def _valid(t):
    return f"<vulnerability>{t}</vulnerability> <trace>0x" + "ab" * 20 + "</trace>"


def test_full_late_step_true_positive():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    cfg = RewardConfig(k=0.01, m=1000.0)
    rb = compute_reward(_valid("reentrancy"), g_obs_reentrancy_full(), step=5000,
                        encoder=enc, references=REFS, cfg=cfg)
    # late step: sigma~1 -> r_graph ~ r_strict(1) + collab(0.3); + r_format(1)
    assert rb.r_strict == 1.0
    assert rb.r_collab == 0.3
    assert rb.r_final > 2.0


def test_full_late_step_true_negative():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    cfg = RewardConfig(k=0.01, m=1000.0)
    rb = compute_reward(_valid("reentrancy"), g_obs_reentrancy_broken(), step=5000,
                        encoder=enc, references=REFS, cfg=cfg)
    assert rb.r_strict == 0.0
    # broken host still shares FFG+CaG views, so collab may fire; key point:
    # strict component is 0, so r_final is well below the true-positive case
    assert rb.r_final < 2.0


def test_full_format_gate_blocks_graph_reward():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    rb = compute_reward("no tags here", g_obs_reentrancy_full(), step=5000,
                        encoder=enc, references=REFS)
    assert rb.r_format == 0.0 and rb.r_graph == 0.0 and rb.r_final == 0.0


def test_full_early_step_is_general_dominated():
    torch.manual_seed(0)
    enc = RGCNEncoder()
    cfg = RewardConfig(k=0.01, m=1000.0)
    rb = compute_reward(_valid("reentrancy"), g_obs_reentrancy_full(), step=0,
                        encoder=enc, references=REFS, cfg=cfg)
    assert rb.sigma < 0.01          # strict barely weighted this early


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
