"""Synthetic spring-mesh dataset for the physics-informed GNN.

Generates a rectangular grid of nodes in 3D with edges between 4-neighbours.
Labels split the grid into two halves along the x-axis — geometry-separable
but not trivial for plain GCNs without positional information.
"""

from __future__ import annotations

import torch

from core.datasets.base import BaseDataset
from core.registry import register


def _grid_mesh(rows: int, cols: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    n = rows * cols
    positions = torch.zeros(n, 3)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            positions[idx] = torch.tensor([float(j), float(i), 0.0])
    # light z-jitter so curvature operators aren't degenerate
    positions[:, 2] = 0.05 * torch.randn(n, generator=g)

    adj = torch.zeros(n, n)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbour = ni * cols + nj
                    adj[idx, neighbour] = 1.0

    return positions, adj


@register("spring_mesh", kind="dataset")
class SpringMeshDataset(BaseDataset):
    """A rectangular 3D mesh split into two classes along the x-axis."""

    def __init__(
        self,
        rows: int = 6,
        cols: int = 6,
        per_class: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__(
            config={"rows": rows, "cols": cols, "per_class": per_class, "seed": seed}
        )
        positions, adj = _grid_mesh(rows, cols, seed=seed)

        labels = (positions[:, 0] >= cols / 2).long()

        g = torch.Generator().manual_seed(seed)
        mask = torch.zeros(labels.size(0), dtype=torch.bool)
        for cls in labels.unique().tolist():
            idx = (labels == cls).nonzero(as_tuple=True)[0]
            perm = idx[torch.randperm(idx.size(0), generator=g)]
            mask[perm[:per_class]] = True

        self.positions = positions
        self.adj = adj
        self.labels = labels
        self.train_mask = mask
        self.x = None  # positions + curvature are used directly

    def _batch(self):
        return self.positions, self.adj, self.x, self.labels, self.train_mask

    def train_batches(self):
        return [self._batch()]

    def val_batches(self):
        return [self._batch()]
