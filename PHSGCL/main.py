from __future__ import annotations

import argparse
import torch

from PHSGCL.models.hycrod import HyCroD, ModelConfig
from PHSGCL.models.prompting import PromptingConfig
from PHSGCL.trainer import TrainConfig, train
from PHSGCL.eval import linear_probe
from PHSGCL.utils.seed import set_seed
from PHSGCL.data import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help=(
            "Dataset name (case-insensitive). Supported: "
            "cora/citeseer/pubmed, amazon-computers/amazon-photos, "
            "coauthor-cs/coauthor-physics, wikics, "
            "webkb-cornell/webkb-texas/webkb-wisconsin, "
            "ogbn-arxiv/ogbn-products."
        ),
    )
    p.add_argument("--data_root", type=str, default="data", help="Root directory for datasets")
    p.add_argument(
        "--split",
        type=str,
        default="public",
        choices=["public", "random", "ogb"],
        help=(
            "Split strategy: public=use built-in masks, "
            "random=random split by ratios, ogb=OGB official split."
        ),
    )
    p.add_argument("--train_ratio", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.8)

    # normalize features (default True), and a flag to disable it
    p.add_argument(
        "--normalize_features",
        action="store_true",
        default=True,
        help="Apply torch_geometric.transforms.NormalizeFeatures (default: True)",
    )
    p.add_argument("--no_normalize_features", action="store_true", help="Disable feature normalization")

    p.add_argument(
        "--to_undirected",
        action="store_true",
        default=False,
        help="Convert graph to undirected (and coalesce edge_index)",
    )

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)

    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--lambda_cv", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=20, help="evaluate every K epochs (freeze encoder and run probes)")

    # augment
    p.add_argument("--node_mask_ratio", type=float, default=0.3)
    p.add_argument("--edge_drop_ratio", type=float, default=0.3)

    # reg
    p.add_argument("--lambda_scat", type=float, default=1.0)
    p.add_argument("--lambda_1hop", type=float, default=1.0)
    p.add_argument("--lambda_mhop", type=float, default=1.0)
    p.add_argument("--k_hop", type=int, default=3)

    # model
    p.add_argument("--gcn_hidden", type=int, default=256)
    p.add_argument("--gcn_out", type=int, default=128)
    p.add_argument("--hgcn_hidden", type=int, default=256)
    p.add_argument("--hgcn_out", type=int, default=128)
    p.add_argument("--gcn_layers", type=int, default=2)
    p.add_argument("--hgcn_layers", type=int, default=2)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--pred_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hyp_c", type=float, default=1.0)

    # diffusion
    p.add_argument("--diff_clusters", type=int, default=10)
    p.add_argument("--diff_noise", type=float, default=0.02)

    # prompting
    p.add_argument("--no_prompt", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    normalize_features = bool(args.normalize_features) and (not args.no_normalize_features)

    data, dataset, meta = load_dataset(
        name=args.dataset,
        root=args.data_root,
        normalize_features=normalize_features,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        to_undirected=args.to_undirected,
    )

    prompt_cfg = PromptingConfig(enable=not args.no_prompt)

    model_cfg = ModelConfig(
        in_dim=meta["num_features"],
        gcn_hidden=args.gcn_hidden,
        gcn_out=args.gcn_out,
        hgcn_hidden=args.hgcn_hidden,
        hgcn_out=args.hgcn_out,
        gcn_layers=args.gcn_layers,
        hgcn_layers=args.hgcn_layers,
        proj_dim=args.proj_dim,
        pred_dim=args.pred_dim,
        dropout=args.dropout,
        tau=args.tau,
        hyp_c=args.hyp_c,
        diff_clusters=args.diff_clusters,
        diff_noise=args.diff_noise,
        prompting=prompt_cfg,
    )

    model = HyCroD(model_cfg)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tau=args.tau,
        lambda_cv=args.lambda_cv,
        gamma=args.gamma,
    )
    train_cfg.eval_interval = args.eval_interval
    train_cfg.checkpoint_path = f"best_checkpoint_{args.dataset}.pt"
    train_cfg.augment.node_mask_ratio = args.node_mask_ratio
    train_cfg.augment.edge_drop_ratio = args.edge_drop_ratio
    train_cfg.reg.lambda_scat = args.lambda_scat
    train_cfg.reg.lambda_1hop = args.lambda_1hop
    train_cfg.reg.lambda_mhop = args.lambda_mhop
    train_cfg.reg.k_hop = args.k_hop

    _ = train(model, data, device=device, cfg=train_cfg)

    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        ei = data.edge_index.to(device)
        out = model.forward_views(x, ei, ei, x, seed=0)
        z = out["h_e1"]  # Euclidean embedding (node-branch)

    acc = linear_probe(
        z,
        data.y.to(device),
        data.train_mask.to(device),
        data.test_mask.to(device),
    )
    print(f"Linear probe accuracy (test): {acc:.4f}")


if __name__ == "__main__":
    main()
