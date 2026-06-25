"""
DG-VDT — graph schema.

Defines the trace-graph data structure that the encoder, the VF2 matcher and
the collaboration bonus all operate on. A trace parser only needs to emit a
TraceGraph; nothing downstream depends on how the trace was produced.

Taxonomy:
  node types  : EOA / Contract / Miner  (address taxonomy)
                + State / Opcode / Branch / Sink  (auxiliary trace nodes)
  edge types  : ETH / CALL / CREATE / SLOAD / SSTORE  (interaction ontology)
  graph views : FFG / CCG / CaG  (each edge belongs to exactly one view)

Edge attributes (used by R_param and the short-address discriminator):
  amount   : float  -- Ether amount on ETH edges
  arg_len  : int    -- ABI-encoded argument byte length on CALL edges
  order    : int    -- call ordering on CaG edges
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import torch
from torch_geometric.data import Data

# ----------------------------------------------------------------------------
# Controlled vocabularies. Order is FIXED — index positions are used as
# one-hot node features and as RGCN relation ids. Do not reorder.
# ----------------------------------------------------------------------------
NODE_TYPES: list[str] = ["EOA", "Contract", "Miner", "State", "Opcode", "Branch", "Sink"]
EDGE_TYPES: list[str] = ["ETH", "CALL", "CREATE", "SLOAD", "SSTORE"]
GRAPH_VIEWS: list[str] = ["FFG", "CCG", "CaG"]

# Which view an edge type lives in by default (paper's construction).
DEFAULT_VIEW: dict[str, str] = {
    "ETH": "FFG",
    "CREATE": "CCG",
    "CALL": "CaG",
    "SLOAD": "CaG",
    "SSTORE": "CaG",
}

NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
EDGE_TYPE_TO_IDX = {t: i for i, t in enumerate(EDGE_TYPES)}

# The three target vulnerability classes, in the exact tag spelling the
# policy must emit inside <vulnerability>...</vulnerability>.
VULN_TYPES: list[str] = ["reentrancy", "short_address", "timestamp"]


@dataclass
class Node:
    nid: Any                      # any hashable id (address string, synthetic id)
    ntype: str                    # one of NODE_TYPES

    def __post_init__(self) -> None:
        if self.ntype not in NODE_TYPE_TO_IDX:
            raise ValueError(f"unknown node type {self.ntype!r}; allowed: {NODE_TYPES}")


@dataclass
class Edge:
    src: Any
    dst: Any
    etype: str                    # one of EDGE_TYPES
    view: str | None = None       # one of GRAPH_VIEWS; defaults from DEFAULT_VIEW
    attrs: dict[str, Any] = field(default_factory=dict)  # amount / arg_len / order ...

    def __post_init__(self) -> None:
        if self.etype not in EDGE_TYPE_TO_IDX:
            raise ValueError(f"unknown edge type {self.etype!r}; allowed: {EDGE_TYPES}")
        if self.view is None:
            self.view = DEFAULT_VIEW[self.etype]
        if self.view not in GRAPH_VIEWS:
            raise ValueError(f"unknown view {self.view!r}; allowed: {GRAPH_VIEWS}")


class TraceGraph:
    """Unified per-window snapshot G_obs = {FFG, CCG, CaG}.

    A single object holds all three views; the `view` field on each edge keeps
    them distinguishable for the multi-graph collaboration bonus.
    """

    def __init__(self, nodes: list[Node] | None = None, edges: list[Edge] | None = None):
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self._ids: set[Any] = set()
        for n in nodes or []:
            self.add_node(n)
        for e in edges or []:
            self.add_edge(e)

    # -- construction --------------------------------------------------------
    def add_node(self, node: Node) -> None:
        if node.nid not in self._ids:
            self.nodes.append(node)
            self._ids.add(node.nid)

    def add_edge(self, edge: Edge) -> None:
        # tolerate edges that reference not-yet-declared endpoints
        for nid in (edge.src, edge.dst):
            if nid not in self._ids:
                self.add_node(Node(nid, "State"))
        self.edges.append(edge)

    def views_present(self) -> set[str]:
        return {e.view for e in self.edges}

    # -- conversions ---------------------------------------------------------
    def to_pyg(self) -> Data:
        """Homogeneous PyG Data for RGCNConv: x (one-hot ntype), edge_index,
        edge_type (relation id)."""
        id_to_local = {n.nid: i for i, n in enumerate(self.nodes)}
        x = torch.zeros(len(self.nodes), len(NODE_TYPES), dtype=torch.float)
        for i, n in enumerate(self.nodes):
            x[i, NODE_TYPE_TO_IDX[n.ntype]] = 1.0
        if self.edges:
            ei = torch.tensor(
                [[id_to_local[e.src] for e in self.edges],
                 [id_to_local[e.dst] for e in self.edges]],
                dtype=torch.long,
            )
            et = torch.tensor([EDGE_TYPE_TO_IDX[e.etype] for e in self.edges], dtype=torch.long)
        else:
            ei = torch.empty(2, 0, dtype=torch.long)
            et = torch.empty(0, dtype=torch.long)
        data = Data(x=x, edge_index=ei, edge_type=et)
        data.num_nodes = len(self.nodes)
        return data

    def to_networkx(self) -> nx.MultiDiGraph:
        """MultiDiGraph for VF2: node attr 'ntype', edge attrs 'etype','view'+attrs."""
        g = nx.MultiDiGraph()
        for n in self.nodes:
            g.add_node(n.nid, ntype=n.ntype)
        for e in self.edges:
            g.add_edge(e.src, e.dst, etype=e.etype, view=e.view, **e.attrs)
        return g

    def __repr__(self) -> str:
        return f"TraceGraph(nodes={len(self.nodes)}, edges={len(self.edges)}, views={sorted(self.views_present())})"
