"""Datasets wired up for the framework GNN models.

Graph workloads in this project are full-batch: a single epoch iterates over
one (features, adj, labels, train_mask) tuple. :class:`KarateClubDataset`
implements exactly that protocol over Zachary's 34-node graph.
"""

from __future__ import annotations

import torch

from backend.core.gnn.model import generate_karate_club
from core.datasets.base import BaseDataset
from core.registry import register


def _build_train_mask(labels: torch.Tensor, per_class: int, seed: int) -> torch.Tensor:
    """Pick ``per_class`` supervised nodes from each class (reproducible)."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(labels.size(0), dtype=torch.bool)
    for cls in labels.unique().tolist():
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.size(0), generator=g)]
        mask[perm[:per_class]] = True
    return mask


@register("karate_club", kind="dataset")
class KarateClubDataset(BaseDataset):
    """Zachary's karate club as a full-batch semi-supervised dataset.

    Parameters
    ----------
    per_class:
        Number of labelled nodes per class used for training. The remaining
        nodes form the validation split.
    seed:
        Seed controlling which nodes become the training set.
    """

    def __init__(self, per_class: int = 4, seed: int = 0) -> None:
        super().__init__(config={"per_class": per_class, "seed": seed})
        adj, features, labels = generate_karate_club()
        train_mask = _build_train_mask(labels, per_class=per_class, seed=seed)

        self.adj = adj
        self.features = features
        self.labels = labels
        self.train_mask = train_mask

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features, self.adj, self.labels, self.train_mask

    def train_batches(self):
        return [self._batch()]

    def val_batches(self):
        return [self._batch()]
