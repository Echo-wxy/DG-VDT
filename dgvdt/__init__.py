"""DG-VDT — graph-guided reward engine."""
from .schema import (
    NODE_TYPES, EDGE_TYPES, GRAPH_VIEWS, VULN_TYPES,
    Node, Edge, TraceGraph,
)
from .encoder import RGCNEncoder, info_nce
from .references import canonical_references, build_median_reference
from .reward import (
    parse_output, cosine_reward, vf2_match, strict_reward,
    sigma, collaboration_bonus, param_penalty,
    compute_reward, RewardConfig, RewardBreakdown, ParsedOutput,
)

__all__ = [
    "NODE_TYPES", "EDGE_TYPES", "GRAPH_VIEWS", "VULN_TYPES",
    "Node", "Edge", "TraceGraph",
    "RGCNEncoder", "info_nce",
    "canonical_references", "build_median_reference",
    "parse_output", "cosine_reward", "vf2_match", "strict_reward",
    "sigma", "collaboration_bonus", "param_penalty",
    "compute_reward", "RewardConfig", "RewardBreakdown", "ParsedOutput",
]
