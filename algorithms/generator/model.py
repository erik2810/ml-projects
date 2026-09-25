"""Framework adapter around :class:`ConditionalGraphGenerator`."""

from __future__ import annotations

from typing import Any

import torch

from backend.core.generator.model import ConditionalGraphGenerator
from core.models.base import BaseModel
from core.registry import register


@register("cond_graph_vae", kind="model")
class ConditionalGenerator(BaseModel):
    """Conditional VAE over flattened-upper-triangle graph adjacency.

    Each batch is an ``(adj_vec, cond)`` tuple — this matches the shape
    produced by :func:`backend.core.generator.model.build_training_set`
    (also wrapped as :class:`~algorithms.generator.dataset.ConditionalGraphsDataset`).
    """

    def __init__(
        self,
        max_nodes: int = 20,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        *,
        beta: float = 1.0,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(
            config={
                "max_nodes": max_nodes,
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
                "beta": beta,
                "lr": lr,
            }
        )
        self.net = ConditionalGraphGenerator(
            max_nodes=max_nodes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            beta=beta,
        )
        self.max_nodes = max_nodes

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        adj_vec, cond = batch
        total, recon, kl = self.net.loss(adj_vec, cond)
        return {"loss": total, "recon": recon, "kl": kl}
