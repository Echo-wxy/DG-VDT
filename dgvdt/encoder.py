"""
DG-VDT — heterogeneous graph encoder Phi.

An extension of the Relational Graph Convolutional Network (RGCN) that
projects a graph into R^256. Relations = EDGE_TYPES (num_relations = 5);
node features = one-hot node type. A two-layer RGCN with mean-pool readout
gives a graph-level embedding. Phi is pre-trained with a graph-contrastive
(InfoNCE) objective and then frozen, so the cosine-similarity reward stays
stationary during policy optimization.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import scatter

from .schema import EDGE_TYPES, NODE_TYPES


class RGCNEncoder(nn.Module):
    """Two-layer RGCN + mean-pool readout -> R^{out_dim} (default 256)."""

    def __init__(
        self,
        in_dim: int = len(NODE_TYPES),
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_relations: int = len(EDGE_TYPES),
        num_bases: int | None = None,
    ):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations, num_bases=num_bases)
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_type, batch=None) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index, edge_type))
        h = self.conv2(h, edge_index, edge_type)          # node embeddings
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        # graph-level readout: mean pool over nodes
        z = scatter(h, batch, dim=0, reduce="mean")
        return z                                           # [num_graphs, out_dim]

    @torch.no_grad()
    def embed(self, data) -> torch.Tensor:
        """Embed a single PyG Data (from TraceGraph.to_pyg()). Returns [out_dim]."""
        self.eval()
        # handle the empty-graph edge case so a degenerate G_obs still yields a vector
        if data.num_nodes == 0:
            return torch.zeros(self.out_dim)
        z = self.forward(data.x, data.edge_index, data.edge_type)
        return z.squeeze(0)


def info_nce(z: torch.Tensor, pos_mask: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Graph-contrastive pre-training loss (temperature tau=0.07).

    z         : [N, d] L2-normalizable embeddings of a batch of graphs.
    pos_mask  : [N, N] boolean, True where i,j are a positive pair
                (same vulnerability label), diagonal ignored.
    Hard-negative mining (top-k nearest dissimilar-class) is applied by the
    sampler that builds the batch, not here.
    """
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / tau                                  # [N, N]
    n = z.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))              # drop self-similarity
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = pos_mask & ~eye
    # mean log-prob over positives for each anchor that has at least one positive
    valid = pos.any(dim=1)
    pos_log_prob = (log_prob * pos).sum(dim=1) / pos.sum(dim=1).clamp(min=1)
    return -(pos_log_prob[valid]).mean()
