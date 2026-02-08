from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def info_nce_loss(p: Tensor, z: Tensor, tau: float = 0.2) -> Tensor:
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    logits = p @ z.t() / tau
    labels = torch.arange(p.size(0), device=p.device)
    return F.cross_entropy(logits, labels)


def symmetric_info_nce(p1: Tensor, z2: Tensor, p2: Tensor, z1: Tensor, tau: float = 0.2) -> Tensor:
    return 0.5 * (info_nce_loss(p1, z2, tau=tau) + info_nce_loss(p2, z1, tau=tau))
