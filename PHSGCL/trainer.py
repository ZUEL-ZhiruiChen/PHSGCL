from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
from torch import Tensor
from torch.optim import Adam
from tqdm import tqdm

from .augment import AugmentConfig, build_two_augmented_views
from .losses import symmetric_info_nce
from .regularization import RegConfig, reg_loss
from .models.hycrod import HyCroD, ModelConfig
from .eval import linear_probe, knn_accuracy, nmi_score


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    tau: float = 0.2
    lambda_cv: float = 0.5   # lambda: balance CV and CN
    gamma: float = 0.1       # gamma: strength of L_reg
    augment: AugmentConfig = AugmentConfig()
    reg: RegConfig = RegConfig()
    eval_interval: int = 20
    checkpoint_path: str = "best_checkpoint.pt"


def train(model: HyCroD, data, device: torch.device, cfg: TrainConfig) -> float:
    """Train the model and periodically evaluate embeddings.

    Returns:
        best_val_acc: best validation accuracy (linear probe) observed.
    """
    model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = 0.0

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Training"):
        model.train()

        ei_drop, x_node_masked = build_two_augmented_views(
            x=x,
            edge_index=edge_index,
            num_nodes=x.size(0),
            cfg=cfg.augment,
        )
        ei_drop = ei_drop.to(device)
        x_node_masked = x_node_masked.to(device)

        out = model.forward_views(
            x=x,
            edge_index=edge_index,
            edge_index_drop=ei_drop,
            x_node_masked=x_node_masked,
            edge_weight=None,
            seed=epoch,
        )

        # Euclidean grid
        l_cv_e = symmetric_info_nce(out["p_e1"], out["zt_e2"], out["p_e2"], out["zt_e1"], tau=cfg.tau)

        # Hyperbolic grid
        l_cv_h = symmetric_info_nce(out["p_h1"], out["zt_h2_diff"], out["p_h2"], out["zt_h1_diff"], tau=cfg.tau)

        l_cv = l_cv_e + l_cv_h
        # Align Euclidean and hyperbolic representations within each view.
        l_cn_v1 = symmetric_info_nce(out["p_e1"], out["zt_h1"], out["p_h1"], out["zt_e1"], tau=cfg.tau)
        l_cn_v2 = symmetric_info_nce(out["p_e2"], out["zt_h2"], out["p_h2"], out["zt_e2"], tau=cfg.tau)
        l_cn = l_cn_v1 + l_cn_v2
        # Apply on both branches and both spaces
        l_reg = (
            reg_loss(out["h_e1"], out["ei_node"], cfg.reg)
            + reg_loss(out["h_h1"], out["ei_node"], cfg.reg)
            + reg_loss(out["h_e2"], out["ei_edge"], cfg.reg)
            + reg_loss(out["h_h2"], out["ei_edge"], cfg.reg)
        )

        # Joint optimization objective 
        l_ssl = cfg.lambda_cv * l_cv + (1.0 - cfg.lambda_cv) * l_cn
        loss = l_ssl + cfg.gamma * l_reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.update_target()

        if epoch % 20 == 0 or epoch == 1:
            tqdm.write(f"Epoch {epoch:03d} | loss={loss.item():.4f} | Lcv={l_cv.item():.4f} | Lcn={l_cn.item():.4f} | Lreg={l_reg.item():.4f}")

        # Periodic evaluation and checkpointing
        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            # Freeze encoder parameters during probe training
            for p in model.online.euc_enc.parameters():
                p.requires_grad_(False)
            for p in model.online.hyp_enc.parameters():
                p.requires_grad_(False)

            model.eval()
            with torch.no_grad():
                out_eval = model.forward_views(x, edge_index, edge_index, x, seed=0)
                z_eval = out_eval["h_e1"]

            # linear probe on val
            val_acc = linear_probe(
                z_eval,
                data.y.to(device),
                data.train_mask.to(device),
                data.val_mask.to(device),
                epochs=200,
                lr=0.01,
            )

            # test acc for reporting
            test_acc = linear_probe(
                z_eval,
                data.y.to(device),
                data.train_mask.to(device),
                data.test_mask.to(device),
                epochs=200,
                lr=0.01,
            )

            # kNN & NMI
            knn_test = knn_accuracy(z_eval, data.y.to(device), data.train_mask.to(device), data.test_mask.to(device), k=5)
            nmi_all = nmi_score(z_eval, data.y.to(device), mask=None)

            tqdm.write(f"Eval @ epoch {epoch}: val_acc={val_acc:.4f}, test_acc={test_acc:.4f}, knn_test={knn_test:.4f}, nmi={nmi_all:.4f}")

            # save best checkpoint by val acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                try:
                    torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc}, cfg.checkpoint_path)
                    tqdm.write(f"Saved best checkpoint to {cfg.checkpoint_path} (val_acc={val_acc:.4f})")
                except Exception as e:
                    tqdm.write(f"Failed to save checkpoint: {e}")

            # unfreeze encoder
            for p in model.online.euc_enc.parameters():
                p.requires_grad_(True)
            for p in model.online.hyp_enc.parameters():
                p.requires_grad_(True)

    return float(best_val_acc)
