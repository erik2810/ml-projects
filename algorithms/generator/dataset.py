"""Datasets for the conditional graph generator."""

from __future__ import annotations

import torch

from backend.core.generator.model import build_training_set
from core.datasets.base import BaseDataset
from core.registry import register


@register("random_graph_bank", kind="dataset")
class ConditionalGraphsDataset(BaseDataset):
    """ER + BA random-graph bank with [nodes, density, clustering] labels.

    Yields mini-batches of ``(adj_vec, cond)`` tensors.
    """

    def __init__(
        self,
        num_graphs: int = 512,
        max_nodes: int = 20,
        batch_size: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            config={
                "num_graphs": num_graphs,
                "max_nodes": max_nodes,
                "batch_size": batch_size,
                "seed": seed,
            }
        )
        if seed is not None:
            torch.manual_seed(seed)
        self.adj_vecs, self.conds = build_training_set(
            num_graphs=num_graphs, max_nodes=max_nodes
        )
        self.batch_size = batch_size

    def _iter_batches(self):
        n = self.adj_vecs.size(0)
        perm = torch.randperm(n)
        for start in range(0, n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            yield self.adj_vecs[idx], self.conds[idx]

    def train_batches(self):
        return list(self._iter_batches())
