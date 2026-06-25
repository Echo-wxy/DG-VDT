"""
DG-VDT — reference attack signatures G*.

Two ways to obtain a signature G*:
  1. canonical_references() -- the three hand-specified signatures
     (RE: 7 nodes/8 edges, SA: 5/5, TD: 6/7), encoding the minimal
     discriminating structure of each vulnerability class.
  2. build_median_reference() -- select, from a pool of confirmed-positive
     graphs of one class, the MEDIAN-size (by node count) connected graph.

Node/edge taxonomy is the one defined in schema.py.
"""
from __future__ import annotations

import statistics

from .schema import Edge, Node, TraceGraph, VULN_TYPES


# ---------------------------------------------------------------------------
# 1. Canonical signatures (minimal discriminating structure per class)
# ---------------------------------------------------------------------------
def _reentrancy_gstar() -> TraceGraph:
    # 7 nodes, 8 edges. Core discriminator: FFG cycle V<->Ac + CaG CALL re-entry.
    nodes = [
        Node("A", "EOA"), Node("V", "Contract"), Node("Ac", "Contract"),
        Node("b0", "State"), Node("b1", "State"), Node("b2", "State"), Node("b3", "State"),
    ]
    edges = [
        Edge("A", "V", "ETH"),            # initial deposit            (FFG)
        Edge("V", "Ac", "ETH"),           # vulnerable withdrawal      (FFG)
        Edge("Ac", "V", "ETH"),           # cycle back -> FFG cycle     (FFG)
        Edge("Ac", "V", "CALL"),          # re-entry invocation        (CaG)
        Edge("V", "b0", "SLOAD"),         # read balance before        (CaG)
        Edge("V", "b1", "SSTORE"),        # write balance after        (CaG)
        Edge("Ac", "b2", "SLOAD"),        # attacker-side ledger        (CaG)
        Edge("Ac", "b3", "SSTORE"),       # attacker-side ledger        (CaG)
    ]
    return TraceGraph(nodes, edges)


def _short_address_gstar() -> TraceGraph:
    # 5 nodes, 5 edges. Discriminator: CALL edge with ABI arg_len < 32 bytes.
    nodes = [
        Node("A", "EOA"), Node("C", "Contract"), Node("R", "EOA"),
        Node("d0", "State"), Node("d1", "State"),
    ]
    edges = [
        Edge("A", "C", "CALL", attrs={"arg_len": 28}),  # truncated transfer  (CaG)
        Edge("C", "d0", "SLOAD"),                        # ABI decode           (CaG)
        Edge("d0", "d1", "SSTORE"),                      # ABI decode           (CaG)
        Edge("A", "C", "ETH"),                           # value attached       (FFG)
        Edge("C", "R", "ETH"),                           # transfer to recipient(FFG)
    ]
    return TraceGraph(nodes, edges)


def _timestamp_gstar() -> TraceGraph:
    # 6 nodes, 7 edges. Discriminator: TIMESTAMP->TRANSFER causal chain (CaG)
    # co-occurring with a miner-account CCG hub (deploy edge).
    nodes = [
        Node("M", "Miner"), Node("L", "Contract"), Node("T", "Opcode"),
        Node("B", "Branch"), Node("S", "Sink"), Node("bal", "State"),
    ]
    edges = [
        Edge("M", "L", "CALL"),       # enter lottery               (CaG)
        Edge("L", "T", "CALL"),       # read TIMESTAMP              (CaG)
        Edge("T", "B", "SLOAD"),      # timestamp feeds branch      (CaG)
        Edge("B", "S", "CALL"),       # branch -> transfer          (CaG)
        Edge("S", "M", "ETH"),        # payout back to miner        (FFG)
        Edge("S", "bal", "SSTORE"),   # balance update              (CaG)
        Edge("M", "L", "CREATE"),     # miner deploy hub            (CCG)
    ]
    return TraceGraph(nodes, edges)


def canonical_references() -> dict[str, TraceGraph]:
    refs = {
        "reentrancy": _reentrancy_gstar(),
        "short_address": _short_address_gstar(),
        "timestamp": _timestamp_gstar(),
    }
    assert set(refs) == set(VULN_TYPES)
    return refs


# ---------------------------------------------------------------------------
# 2. Median-size construction
# ---------------------------------------------------------------------------
def build_median_reference(positive_pool: list[TraceGraph]) -> TraceGraph:
    """Pick the median-size (by node count) graph from a pool of confirmed
    positive graphs of one vulnerability class. Ties resolve to the lower
    index. The median is a breakdown-point-robust size selector, more stable to
    outlier graph sizes than the mean.
    """
    if not positive_pool:
        raise ValueError("empty positive pool")
    sizes = [len(g.nodes) for g in positive_pool]
    target = statistics.median_low(sizes)
    # among graphs whose size == median_low, return the first
    for g in positive_pool:
        if len(g.nodes) == target:
            return g
    # fallback (shouldn't happen): closest to median
    return min(positive_pool, key=lambda g: abs(len(g.nodes) - target))
