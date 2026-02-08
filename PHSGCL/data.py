from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import coalesce, to_undirected


def _canonical_name(name: str) -> str:
    """Normalize dataset name (case-insensitive, hyphen/underscore tolerant)."""
    n = name.strip().lower()
    n = n.replace("_", "-")
    while "--" in n:
        n = n.replace("--", "-")
    return n


def _ensure_y_node_level(data: Data) -> Data:
    """Ensure data.y is a 1D LongTensor of shape [num_nodes]."""
    if not hasattr(data, "y") or data.y is None:
        raise ValueError("Loaded dataset has no labels (data.y is missing).")
    y = data.y
    if isinstance(y, (list, tuple)):
        y = torch.tensor(y)
    if y.dim() > 1:
        y = y.view(-1)
    if y.dtype != torch.long:
        y = y.to(torch.long)
    data.y = y
    return data


def _random_split_masks(
    num_nodes: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio+val_ratio+test_ratio must sum to 1.0, got "
            f"{train_ratio}+{val_ratio}+{test_ratio}"
        )

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(num_nodes, generator=g)

    n_train = int(round(train_ratio * num_nodes))
    n_val = int(round(val_ratio * num_nodes))
    n_train = min(n_train, num_nodes)
    n_val = min(n_val, num_nodes - n_train)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _idx_to_mask(num_nodes: int, idx: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = True
    return mask


def _postprocess_graph(
    data: Data,
    *,
    normalize_features: bool,
    to_undirected_flag: bool,
) -> Data:
    if normalize_features:
        data = NormalizeFeatures()(data)

    if data.edge_index is None:
        raise ValueError("Loaded dataset has no edge_index.")
    data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)

    if to_undirected_flag:
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)

    return data


def load_dataset(
    name: str,
    root: str,
    normalize_features: bool = True,
    split: str = "public",
    **kwargs: Any,
) -> Tuple[Data, Any, Dict[str, Any]]:
    """Unified dataset loader.

    Args:
        name: Dataset name (case-insensitive). Supported (aliases):
            - Planetoid: cora, citeseer, pubmed
            - Amazon: amazon-computers, amazon-photos
            - Coauthor: coauthor-cs, coauthor-physics
            - WikiCS: wikics
            - WebKB: webkb-cornell, webkb-texas, webkb-wisconsin
            - OGB: ogbn-arxiv, ogbn-products
        root: Root directory for datasets.
        normalize_features: Whether to apply NormalizeFeatures(). Default True.
        split: One of {"public", "random", "ogb"}.
        **kwargs:
            - train_ratio, val_ratio, test_ratio, seed (for split="random")
            - to_undirected (bool)

    Returns:
        data: Data with required fields:
            x, edge_index, y, train_mask, val_mask, test_mask, num_nodes
        dataset: underlying dataset object
        meta: dict with at least:
            num_features, num_classes, is_undirected, has_edge_attr, name, split, task_type
    """
    n = _canonical_name(name)
    split = split.strip().lower()
    if split not in {"public", "random", "ogb"}:
        raise ValueError(f"Unsupported split='{split}'. Choose from public/random/ogb.")

    to_undirected_flag = bool(kwargs.pop("to_undirected", False))


    planetoid_map = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}

    amazon_map = {
        "amazon-computers": "Computers",
        "amazon-photos": "Photo",
        "amazon-photo": "Photo",
        "amazon-photographs": "Photo",
    }

    coauthor_map = {
        "coauthor-cs": "CS",
        "coauthor-physics": "Physics",
        "coauthor-phys": "Physics",
    }

    wikics_set = {"wikics", "wiki-cs"}

    webkb_map = {
        "webkb-cornell": "Cornell",
        "webkb-texas": "Texas",
        "webkb-wisconsin": "Wisconsin",
        "cornell": "Cornell",
        "texas": "Texas",
        "wisconsin": "Wisconsin",
    }

    ogb_set = {"ogbn-arxiv", "ogbn-products"}


    if n in planetoid_map:
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(root=root, name=planetoid_map[n])
        data = dataset[0]
    elif n in amazon_map:
        from torch_geometric.datasets import Amazon

        dataset = Amazon(root=root, name=amazon_map[n])
        data = dataset[0]
    elif n in coauthor_map:
        from torch_geometric.datasets import Coauthor

        dataset = Coauthor(root=root, name=coauthor_map[n])
        data = dataset[0]
    elif n in wikics_set:
        from torch_geometric.datasets import WikiCS

        dataset = WikiCS(root=root)
        data = dataset[0]
    elif n in webkb_map:
        from torch_geometric.datasets import WebKB

        dataset = WebKB(root=root, name=webkb_map[n])
        data = dataset[0]
    elif n in ogb_set:
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except Exception as e:
            raise ImportError(
                "OGB is required for ogbn-* datasets. Install with: pip install ogb"
            ) from e

        dataset = PygNodePropPredDataset(name=n, root=root)
        data = dataset[0]
        if hasattr(data, "y") and data.y is not None:
            data.y = data.y.view(-1)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: "
            "cora/citeseer/pubmed, amazon-computers/amazon-photos, "
            "coauthor-cs/coauthor-physics, wikics, "
            "webkb-cornell/webkb-texas/webkb-wisconsin, ogbn-arxiv/ogbn-products."
        )


    if getattr(data, "x", None) is None:
        raise ValueError("Loaded dataset has no node features (data.x is missing).")
    if getattr(data, "num_nodes", None) is None:
        data.num_nodes = data.x.size(0)

    data = _ensure_y_node_level(data)


    if split == "public":
        for k in ["train_mask", "val_mask", "test_mask"]:
            if not hasattr(data, k) or getattr(data, k) is None:
                raise ValueError(
                    f"split='public' requires data.{k} to exist for dataset '{name}'. "
                    "Use --split random (or ogb for ogbn-*) instead."
                )
    elif split == "random":
        train_ratio = float(kwargs.pop("train_ratio", 0.1))
        val_ratio = float(kwargs.pop("val_ratio", 0.1))
        test_ratio = float(kwargs.pop("test_ratio", 0.8))
        seed = int(kwargs.pop("seed", 42))
        tr, va, te = _random_split_masks(
            data.num_nodes,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        data.train_mask, data.val_mask, data.test_mask = tr, va, te
    elif split == "ogb":
        if n not in ogb_set:
            raise ValueError("split='ogb' is only valid for ogbn-* datasets.")
        idx = dataset.get_idx_split()
        train_idx = idx["train"]
        valid_idx = idx.get("valid", idx.get("val"))
        test_idx = idx["test"]
        data.train_mask = _idx_to_mask(data.num_nodes, train_idx)
        data.val_mask = _idx_to_mask(data.num_nodes, valid_idx)
        data.test_mask = _idx_to_mask(data.num_nodes, test_idx)


    data = _postprocess_graph(
        data,
        normalize_features=normalize_features,
        to_undirected_flag=to_undirected_flag,
    )

    is_undirected = bool(getattr(data, "is_undirected", lambda: False)()) if hasattr(data, "is_undirected") else False
    has_edge_attr = getattr(data, "edge_attr", None) is not None
    num_classes = int(getattr(dataset, "num_classes", int(data.y.max().item() + 1)))

    meta: Dict[str, Any] = {
        "name": n,
        "split": split,
        "task_type": "node_classification",
        "num_features": int(data.x.size(-1)),
        "num_classes": num_classes,
        "is_undirected": is_undirected,
        "has_edge_attr": bool(has_edge_attr),
    }

    return data, dataset, meta
