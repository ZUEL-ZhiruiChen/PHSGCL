from __future__ import annotations

from dataclasses import dataclass
import copy
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field

from .gcn import GCNEncoder
from .hyperbolic import HGCNEncoder, hyp_to_tangent0
from .mlp import MLP
from .prompting import GraphSelectivePrompting, PromptingConfig


@dataclass
class ModelConfig:
    in_dim: int
    gcn_hidden: int = 256
    gcn_out: int = 128
    hgcn_hidden: int = 256
    hgcn_out: int = 128
    gcn_layers: int = 2
    hgcn_layers: int = 2
    proj_dim: int = 128
    pred_dim: int = 128
    dropout: float = 0.2
    tau: float = 0.2
    ema: float = 0.99
    hyp_c: float = 1.0
    # diffusion aug
    diff_clusters: int = 10
    diff_noise: float = 0.02
    # prompting
    prompting: PromptingConfig = field(default_factory=PromptingConfig)


class OnlineTarget(nn.Module):
    """Container for online & target encoders and heads."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        in2 = 2 * cfg.in_dim  

        self.euc_enc = GCNEncoder(in2, cfg.gcn_hidden, cfg.gcn_out, num_layers=cfg.gcn_layers, dropout=cfg.dropout)
        self.hyp_enc = HGCNEncoder(in2, cfg.hgcn_hidden, cfg.hgcn_out, num_layers=cfg.hgcn_layers, c=cfg.hyp_c)

        self.euc_proj = MLP(cfg.gcn_out, cfg.proj_dim, cfg.proj_dim, num_layers=2, dropout=cfg.dropout)
        self.hyp_proj = MLP(cfg.hgcn_out, cfg.proj_dim, cfg.proj_dim, num_layers=2, dropout=cfg.dropout)

        self.euc_pred = MLP(cfg.proj_dim, cfg.pred_dim, cfg.proj_dim, num_layers=2, dropout=cfg.dropout)
        self.hyp_pred = MLP(cfg.proj_dim, cfg.pred_dim, cfg.proj_dim, num_layers=2, dropout=cfg.dropout)

    def forward_euc(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        h = self.euc_enc(x, edge_index, edge_weight=edge_weight)
        z = self.euc_proj(h)
        p = self.euc_pred(z)
        return h, z, p

    def forward_hyp(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        h_h = self.hyp_enc(x, edge_index, edge_weight=edge_weight)
        h = hyp_to_tangent0(h_h, c=self.hyp_enc.c)  
        z = self.hyp_proj(h)
        p = self.hyp_pred(z)
        return h, z, p


class HyCroD(nn.Module):
    """Strict re-implementation aligned with Sections 3.2-3.5 (methodology doc)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.prompting = GraphSelectivePrompting(cfg.in_dim, cfg.prompting)

        self.online = OnlineTarget(cfg)
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self) -> None:
        m = self.cfg.ema
        for p_o, p_t in zip(self.online.parameters(), self.target.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

    def forward_views(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_index_drop: Tensor,
        x_node_masked: Tensor,
        edge_weight: Tensor | None = None,
        seed: int = 0,
    ) -> dict:

        out = self.prompting(
            x=x,
            x_node_masked=x_node_masked,
            edge_index_orig=edge_index,
            edge_index_dropped=edge_index_drop,
            edge_weight_orig=edge_weight,
        )

        # Node-branch view 
        x_node = out["x_node"]
        ei_node = out["edge_index_node"]
        ew_node = out["edge_weight_node"]

        # Edge-branch view 
        x_edge = out["x_edge"]
        ei_edge = out["edge_index_edge"]
        ew_edge = out["edge_weight_edge"]

        # dual-space encoders on both branches
        h_e1, z_e1, p_e1 = self.online.forward_euc(x_node, ei_node, edge_weight=ew_node)
        h_h1, z_h1, p_h1 = self.online.forward_hyp(x_node, ei_node, edge_weight=ew_node)

        h_e2, z_e2, p_e2 = self.online.forward_euc(x_edge, ei_edge, edge_weight=ew_edge)
        h_h2, z_h2, p_h2 = self.online.forward_hyp(x_edge, ei_edge, edge_weight=ew_edge)

        # Target network (no grad)
        with torch.no_grad():
            ht_e1, zt_e1, _ = self.target.forward_euc(x_node, ei_node, edge_weight=ew_node)
            ht_h1, zt_h1, _ = self.target.forward_hyp(x_node, ei_node, edge_weight=ew_node)

            ht_e2, zt_e2, _ = self.target.forward_euc(x_edge, ei_edge, edge_weight=ew_edge)
            ht_h2, zt_h2, _ = self.target.forward_hyp(x_edge, ei_edge, edge_weight=ew_edge)

            zt_h1_diff = zt_h1
            zt_h2_diff = zt_h2

        return dict(
            # online preds (for InfoNCE)
            p_e1=p_e1, p_h1=p_h1, p_e2=p_e2, p_h2=p_h2,
            # target projections
            zt_e1=zt_e1, zt_h1=zt_h1, zt_e2=zt_e2, zt_h2=zt_h2,
            # embeddings for regularization
            h_e1=h_e1, h_h1=h_h1, h_e2=h_e2, h_h2=h_h2,
            zt_h1_diff=zt_h1_diff, zt_h2_diff=zt_h2_diff,
            # aux prompting weights
            aux=out["aux"],
            # edge indices for regularization
            ei_node=ei_node, ei_edge=ei_edge,
        )
