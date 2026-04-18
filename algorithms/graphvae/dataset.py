"""Datasets for the Graph VAE / diffusion models."""

from __future__ import annotations

from backend.core.graphvae.model import generate_social_skeletons
from core.datasets.base import BaseDataset
from core.registry import register


@register("social_skeletons", kind="dataset")
class SocialSkeletonsDataset(BaseDataset):
    """Mixture of Watts-Strogatz and Barabasi-Albert graphs.

    Each batch is a single ``(features, adj)`` tuple; an epoch iterates over
    the full collection. Sizes are chosen to stay under ``max_nodes`` so that
    zero-padding inside the model is well-defined.
    """

    def __init__(
        self,
        num_graphs: int = 50,
        sizes: list[int] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(config={"num_graphs": num_graphs, "sizes": sizes, "seed": seed})
        if seed is not None:
            import torch

            torch.manual_seed(seed)
        self._graphs = generate_social_skeletons(num_graphs=num_graphs, sizes=sizes)

    def train_batches(self):
        return list(self._graphs)
