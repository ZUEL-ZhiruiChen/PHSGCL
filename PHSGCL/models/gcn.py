from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """L-layer GCN encoder (Euclidean branch)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        dims = [in_dim] + ([hidden_dim] * (num_layers - 1)) + [out_dim]
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i+1], cached=False, normalize=True) for i in range(num_layers)])
        self.acts = nn.ModuleList([nn.PReLU() for _ in range(num_layers - 1)])

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                h = self.acts[i](h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        return h
