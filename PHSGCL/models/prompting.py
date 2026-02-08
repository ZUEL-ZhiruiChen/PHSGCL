from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PromptingConfig:
    """Graph Selective Prompting module config."""
    enable: bool = True
    # initialization scale for prompt parameters
    node_prompt_init: float = 0.01
    edge_prompt_init: float = 0.01


class NodeFeaturePrompting(nn.Module):
    r"""Node Feature Prompting.

    Eq.(1): x_i^p = x_i + alpha_i * p_n
    Eq.(2): alpha_i = sigmoid( W_n^T x_i + b_n )
    where:
        p_n \in R^d is a learnable prompt vector,
        W_n \in R^d, b_n \in R are learnable parameters.
    """

    def __init__(self, d: int, init: float = 0.01):
        super().__init__()
        self.p_n = nn.Parameter(torch.randn(d) * init)
        self.W_n = nn.Parameter(torch.randn(d) * init)
        self.b_n = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        alpha = torch.sigmoid(x @ self.W_n + self.b_n)  # [N]
        x_p = x + alpha.unsqueeze(-1) * self.p_n.unsqueeze(0)
        return x_p, alpha


class EdgeStructurePrompting(nn.Module):
    r"""Edge Structure Prompting.

    Eq.(3): A_ij^p = A_ij * (1 + beta_ij * p_e)
    Eq.(4): beta_ij = sigmoid( W_e^T [x_i || x_j] + b_e )

    For unweighted graphs, A_ij = 1 on observed edges.
    p_e is implemented as a learnable scalar.
    """

    def __init__(self, d: int, init: float = 0.01):
        super().__init__()
        self.p_e = nn.Parameter(torch.randn(()) * init)     # scalar
        self.W_e = nn.Parameter(torch.randn(2 * d) * init)  # vector in R^{2d}
        self.b_e = nn.Parameter(torch.zeros(()))            # scalar

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        pair = torch.cat([x[src], x[dst]], dim=-1)           # [E, 2d]
        beta = torch.sigmoid(pair @ self.W_e + self.b_e)     # [E]
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        w_p = edge_weight * (1.0 + beta * self.p_e)          # [E]
        return w_p, beta


class GraphSelectivePrompting(nn.Module):
    """Full Graph Selective Prompting module.

    Given original graph G0=(V,E) with (X,A), we build:
      - G1: node-mask augmented view (feature corruption)
      - G2: edge-mask augmented view (structure corruption)

    Then we compute:
      - node feature prompting: X^p (Eq.1-2)
      - edge structure prompting: A^p (Eq.3-4) (implemented via edge weights)

    Finally we concatenate augmented data with prompted data to form
    the final inputs for the two branches:

      Node-branch input features:   X_node = [X_masked || X_prompted]
      Edge-branch input structure:  E_edge = E_drop  ⊕  E_prompted (concatenation of edge lists)
                                   with weights [1 || w_prompted]
      Edge-branch features are also concatenated to match dimensions:
                                   X_edge = [X || X]  (structure prompting acts on topology)

    This matches the text description that masked-enhanced data are concatenated
    with prompted data to build the final graph inputs.
    """

    def __init__(self, d: int, cfg: PromptingConfig):
        super().__init__()
        self.cfg = cfg
        self.node = NodeFeaturePrompting(d, init=cfg.node_prompt_init)
        self.edge = EdgeStructurePrompting(d, init=cfg.edge_prompt_init)

    def forward(
        self,
        x: Tensor,
        x_node_masked: Tensor,
        edge_index_orig: Tensor,
        edge_index_dropped: Tensor,
        edge_weight_orig: Tensor | None = None,
    ) -> dict:
        if not self.cfg.enable:
            # fall back to simple two-view setup with feature concatenation identity
            x_node = torch.cat([x_node_masked, x], dim=-1)
            x_edge = torch.cat([x, x], dim=-1)
            ew_drop = torch.ones(edge_index_dropped.size(1), device=x.device)
            return dict(
                x_node=x_node,
                edge_index_node=edge_index_orig,
                edge_weight_node=edge_weight_orig,
                x_edge=x_edge,
                edge_index_edge=edge_index_dropped,
                edge_weight_edge=ew_drop,
                aux=dict(alpha=None, beta=None),
            )

        x_p, alpha = self.node(x)
        x_node = torch.cat([x_node_masked, x_p], dim=-1)

        w_p, beta = self.edge(x, edge_index_orig, edge_weight=edge_weight_orig)
        # concatenate dropped edges and prompted edges
        ew_drop = torch.ones(edge_index_dropped.size(1), device=x.device)
        edge_index_edge = torch.cat([edge_index_dropped, edge_index_orig], dim=1)
        edge_weight_edge = torch.cat([ew_drop, w_p], dim=0)

        x_edge = torch.cat([x, x], dim=-1)

        return dict(
            x_node=x_node,
            edge_index_node=edge_index_orig,
            edge_weight_node=edge_weight_orig,
            x_edge=x_edge,
            edge_index_edge=edge_index_edge,
            edge_weight_edge=edge_weight_edge,
            aux=dict(alpha=alpha, beta=beta),
        )
