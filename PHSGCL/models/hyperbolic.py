from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
import geoopt
import geoopt.manifolds.stereographic.math as pmath
from torch_scatter import scatter_add


def _k_tensor(c: float | Tensor, ref: Tensor) -> Tensor:
    k = -c
    if isinstance(k, Tensor):
        return k.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(k, device=ref.device, dtype=ref.dtype)


class HyperbolicLinear(nn.Module):
    """Hyperbolic linear layer using Möbius matvec + Möbius bias add."""

    def __init__(self, in_dim: int, out_dim: int, c: float):
        super().__init__()
        self.c = c
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x_h: Tensor) -> Tensor:
        k = _k_tensor(self.c, x_h)
        # Möbius matrix-vector multiplication
        y = pmath.mobius_matvec(self.W, x_h, k=k)
        # Möbius bias addition: map bias to manifold then add
        b_h = pmath.expmap0(self.b, k=k)
        y = pmath.mobius_add(y, b_h, k=k)
        return y


class HGCNLayer(nn.Module):
    """A single HGCN layer with tangent-space aggregation + Möbius activation."""

    def __init__(self, in_dim: int, out_dim: int, c: float):
        super().__init__()
        self.c = c
        self.lin = HyperbolicLinear(in_dim, out_dim, c=c)
        self.act = nn.PReLU()

    def forward(self, x_h: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        k = _k_tensor(self.c, x_h)
        # hyperbolic linear transform
        h = self.lin(x_h)

        # aggregate in tangent space at origin
        h_t = pmath.logmap0(h, k=k)  # [N, d]
        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            w = torch.ones(edge_index.size(1), device=h.device)
        else:
            w = edge_weight
        msg = h_t[src] * w.unsqueeze(-1)
        agg = scatter_add(msg, dst, dim=0, dim_size=h_t.size(0))
        # add self-loop contribution (identity) as common in GCN-style normalization
        agg = agg + h_t

        # back to manifold
        h2 = pmath.expmap0(agg, k=k)

        # Möbius activation
        h2_t = pmath.logmap0(h2, k=k)
        h2_t = self.act(h2_t)
        h2 = pmath.expmap0(h2_t, k=k)
        return pmath.project(h2, k=k)


class HGCNEncoder(nn.Module):
    """Stacked HGCN encoder."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, c: float):
        super().__init__()
        assert num_layers >= 1
        self.c = c
        dims = [in_dim] + ([hidden_dim] * (num_layers - 1)) + [out_dim]
        self.layers = nn.ModuleList([HGCNLayer(dims[i], dims[i+1], c=c) for i in range(num_layers)])

    def forward(self, x_euc: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        k = _k_tensor(self.c, x_euc)
        # map Euclidean features to hyperbolic (origin expmap)
        x_h = pmath.expmap0(x_euc, k=k)
        x_h = pmath.project(x_h, k=k)
        h = x_h
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight=edge_weight)
        return h  # hyperbolic points


def hyp_to_tangent0(x_h: Tensor, c: float) -> Tensor:
    k = _k_tensor(c, x_h)
    return pmath.logmap0(x_h, k=k)
