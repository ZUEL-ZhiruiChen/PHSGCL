from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


def _prepare_masks_and_labels(z: Tensor, y: Tensor, mask: Tensor):
    # normalize common shapes: mask may be 2D, y may be [N,1]
    if mask is not None and mask.dim() > 1:
        mask = mask[:, 0]
    if y.dim() > 1 and y.shape[1] == 1:
        y = y.squeeze(1)
    return z, y, mask


def linear_probe(z: Tensor, y: Tensor, train_mask: Tensor, eval_mask: Tensor, epochs: int = 200, lr: float = 0.01) -> float:
    """Train a linear classifier on `train_mask` and evaluate on `eval_mask`.

    Returns accuracy on eval_mask.
    """
    z = z.detach()
    device = z.device

    z, y, train_mask = _prepare_masks_and_labels(z, y, train_mask)
    _, y, eval_mask = _prepare_masks_and_labels(z, y, eval_mask)

    clf = nn.Linear(z.size(1), int(y.max().item()) + 1).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        clf.train()
        logits = clf(z[train_mask])
        loss = F.cross_entropy(logits, y[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()

    clf.eval()
    pred = clf(z[eval_mask]).argmax(dim=-1).cpu().numpy()
    acc = accuracy_score(y[eval_mask].cpu().numpy(), pred)
    return float(acc)


def knn_accuracy(z: Tensor, y: Tensor, train_mask: Tensor, eval_mask: Tensor, k: int = 5) -> float:
    """k-NN classifier accuracy: fit on train_mask, evaluate on eval_mask."""
    import numpy as _np

    z_np = z.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # normalize masks to numpy boolean/index arrays
    if isinstance(train_mask, torch.Tensor):
        train_mask_np = train_mask.detach().cpu().numpy()
    else:
        train_mask_np = _np.asarray(train_mask)
    if isinstance(eval_mask, torch.Tensor):
        eval_mask_np = eval_mask.detach().cpu().numpy()
    else:
        eval_mask_np = _np.asarray(eval_mask)

    if train_mask_np.ndim > 1:
        train_mask_np = train_mask_np[:, 0]
    if eval_mask_np.ndim > 1:
        eval_mask_np = eval_mask_np[:, 0]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(z_np[train_mask_np.astype(bool)], y_np[train_mask_np.astype(bool)])
    pred = knn.predict(z_np[eval_mask_np.astype(bool)])
    return float(accuracy_score(y_np[eval_mask_np.astype(bool)], pred))


def nmi_score(z: Tensor, y: Tensor, mask: Tensor | None = None, n_clusters: int | None = None) -> float:
    """Compute NMI between KMeans clusters on z and ground-truth labels y.

    If mask is provided, only use masked nodes. By default n_clusters = num classes.
    """
    z_np = z.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    if mask is not None and mask.dim() > 1:
        mask = mask[:, 0]
    if mask is not None:
        z_np = z_np[mask]
        y_np = y_np[mask]

    if n_clusters is None:
        n_clusters = int(y_np.max()) + 1

    km = KMeans(n_clusters=n_clusters, n_init=10)
    preds = km.fit_predict(z_np)
    return float(normalized_mutual_info_score(y_np, preds))