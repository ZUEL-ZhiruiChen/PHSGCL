from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch_geometric.utils import to_undirected


@dataclass
class AugmentConfig:
    """Graph data augmentation to form two views.

    - Node-mask view: mask a ratio of node features (feature corruption).
    - Edge-mask view: drop a ratio of edges (structure corruption).
    """
    node_mask_ratio: float = 0.3
    edge_drop_ratio: float = 0.3
    undirected: bool = True


def node_feature_mask(x: Tensor, mask_ratio: float, mask_value: float = 0.0) -> Tensor:
    """Mask node features by zeroing out all feature dims for a subset of nodes."""
    if mask_ratio <= 0:
        return x
    n = x.size(0)
    m = int(round(n * mask_ratio))
    if m <= 0:
        return x
    idx = torch.randperm(n, device=x.device)[:m]
    x2 = x.clone()
    x2[idx] = mask_value
    return x2


def edge_drop(edge_index: Tensor, drop_ratio: float, num_nodes: int, undirected: bool = True) -> Tensor:
    """Randomly drop a ratio of edges."""
    if drop_ratio <= 0:
        return edge_index
    e = edge_index.size(1)
    keep = int(round(e * (1.0 - drop_ratio)))
    keep = max(1, keep)
    perm = torch.randperm(e, device=edge_index.device)[:keep]
    ei = edge_index[:, perm]
    if undirected:
        ei = to_undirected(ei, num_nodes=num_nodes)
    return ei


def build_two_augmented_views(
    x: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    cfg: AugmentConfig,
) -> Tuple[Tensor, Tensor]:
    """Return (edge_index_edge_masked, x_node_masked)."""
    x_node_masked = node_feature_mask(x, cfg.node_mask_ratio)
    ei_edge_masked = edge_drop(edge_index, cfg.edge_drop_ratio, num_nodes=num_nodes, undirected=cfg.undirected)
    return ei_edge_masked, x_node_masked
