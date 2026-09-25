"""Synthetic tree-structured dataset for hyperbolic models.

Hyperbolic embeddings shine on data with tree-like / hierarchical structure
because the volume of a ball in hyperbolic space grows exponentially with
radius — just like the node count in a balanced tree. This dataset produces
a balanced rooted tree labelled by depth-mod-k, which is trivially separable
with the right geometry but near-impossible for flat Euclidean encoders.
"""

from __future__ import annotations

import torch

from core.datasets.base import BaseDataset
from core.registry import register


def _balanced_tree(depth: int, branching: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a balanced rooted tree. Returns (adj, depths)."""
    nodes = [0]
    parents = [-1]
    depths = [0]
    frontier = [0]
    for d in range(depth):
        next_frontier = []
        for parent in frontier:
            for _ in range(branching):
                new_id = len(nodes)
                nodes.append(new_id)
                parents.append(parent)
                depths.append(d + 1)
                next_frontier.append(new_id)
        frontier = next_frontier

    n = len(nodes)
    adj = torch.zeros(n, n)
    for child, parent in enumerate(parents):
        if parent >= 0:
            adj[child, parent] = 1.0
            adj[parent, child] = 1.0
    return adj, torch.tensor(depths, dtype=torch.long)


@register("tree_graph", kind="dataset")
class TreeGraphDataset(BaseDataset):
    """Balanced rooted tree with depth-based node labels.

    Emits ``(features, adj, labels, train_mask)`` — the standard node
    classification batch shape used across the framework.
    """

    def __init__(
        self,
        depth: int = 3,
        branching: int = 3,
        num_classes: int = 2,
        per_class: int = 3,
        seed: int = 0,
    ) -> None:
        super().__init__(
            config={
                "depth": depth,
                "branching": branching,
                "num_classes": num_classes,
                "per_class": per_class,
                "seed": seed,
            }
        )
        adj, depths = _balanced_tree(depth, branching)
        labels = depths % num_classes

        # one-hot degree features
        deg = adj.sum(dim=1).long()
        max_deg = deg.max().item()
        features = torch.nn.functional.one_hot(deg, num_classes=max_deg + 1).float()

        g = torch.Generator().manual_seed(seed)
        mask = torch.zeros(labels.size(0), dtype=torch.bool)
        for cls in labels.unique().tolist():
            idx = (labels == cls).nonzero(as_tuple=True)[0]
            perm = idx[torch.randperm(idx.size(0), generator=g)]
            mask[perm[:per_class]] = True

        self.adj = adj
        self.features = features
        self.labels = labels
        self.train_mask = mask

    def _batch(self):
        return self.features, self.adj, self.labels, self.train_mask

    def train_batches(self):
        return [self._batch()]

    def val_batches(self):
        return [self._batch()]
