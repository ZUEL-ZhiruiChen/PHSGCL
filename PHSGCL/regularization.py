from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class RegConfig:
    lambda_scat: float = 1.0
    lambda_1hop: float = 1.0
    lambda_mhop: float = 1.0
    k_hop: int = 3


def compute_center(h: Tensor) -> Tensor:
    return h.mean(dim=0, keepdim=True)


def scattering_loss(h: Tensor) -> Tensor:
    c = compute_center(h)
    dist = (h - c).pow(2).sum(dim=-1).sqrt()  # Euclidean distance
    return -dist.mean()


def one_hop_loss(h: Tensor, edge_index: Tensor) -> Tensor:
    src, dst = edge_index[0], edge_index[1]
    return (h[src] - h[dst]).pow(2).sum(dim=-1).mean()


def multihop_strength(edge_index: Tensor, num_nodes: int, k: int) -> Tensor:
    device = edge_index.device
    # build adjacency in dense form (Planetoid graphs are small; strict replication uses matrix power)
    A = torch.zeros((num_nodes, num_nodes), device=device)
    A[edge_index[0], edge_index[1]] = 1.0
    deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    P = A / deg
    S = P
    for _ in range(k - 1):
        S = S @ P
    return S


def multi_hop_loss(h: Tensor, edge_index: Tensor, k: int) -> Tensor:
    n = h.size(0)
    S = multihop_strength(edge_index, n, k=k)  # [N,N]
    # avoid diagonal
    S = S * (1.0 - torch.eye(n, device=h.device))
    # normalize weights
    W = S / (S.sum() + 1e-12)
    # pairwise squared distances
    # (h_i - h_j)^2 = ||h||^2_i + ||h||^2_j - 2 h_i·h_j
    hh = (h * h).sum(dim=1, keepdim=True)  # [N,1]
    dist2 = hh + hh.t() - 2.0 * (h @ h.t())
    return (W * dist2).sum()


def reg_loss(h: Tensor, edge_index: Tensor, cfg: RegConfig) -> Tensor:
    l_s = scattering_loss(h)
    l_1 = one_hop_loss(h, edge_index)
    l_m = multi_hop_loss(h, edge_index, k=cfg.k_hop) if cfg.k_hop >= 2 else torch.zeros((), device=h.device)
    return cfg.lambda_scat * l_s + cfg.lambda_1hop * l_1 + cfg.lambda_mhop * l_m
