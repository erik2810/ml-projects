"""Framework-native adapters for GraphVAE and discrete graph diffusion."""

from __future__ import annotations

from typing import Any

import torch

from backend.core.graphvae.model import DenoisingDiffusion, GraphVAE
from core.models.base import BaseModel
from core.registry import register


def _pad(adj: torch.Tensor, features: torch.Tensor, max_nodes: int):
    n = adj.size(0)
    if n >= max_nodes:
        return adj[:max_nodes, :max_nodes], features[:max_nodes]
    adj_pad = torch.zeros(max_nodes, max_nodes, device=adj.device)
    adj_pad[:n, :n] = adj
    feat_pad = torch.zeros(max_nodes, features.size(1), device=features.device)
    feat_pad[:n] = features
    return adj_pad, feat_pad


@register("graph_vae", kind="model")
class GraphVAEModel(BaseModel):
    """GraphVAE driven by :class:`core.Trainer`.

    Each batch is a ``(features, adj)`` tuple corresponding to a single graph.
    The batch is zero-padded to ``max_nodes`` internally.
    """

    def __init__(
        self,
        max_nodes: int = 20,
        node_feat_dim: int = 11,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        *,
        beta: float = 1.0,
        use_matching: bool = False,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(
            config={
                "max_nodes": max_nodes,
                "node_feat_dim": node_feat_dim,
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
                "beta": beta,
                "use_matching": use_matching,
                "lr": lr,
            }
        )
        self.vae = GraphVAE(
            max_nodes=max_nodes,
            node_feat_dim=node_feat_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            use_matching=use_matching,
        )
        self.beta = beta
        self.max_nodes = max_nodes

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        features, adj = batch
        adj_pad, feat_pad = _pad(adj, features, self.max_nodes)
        adj_hat, _, mu, logvar = self.vae(feat_pad, adj_pad)
        losses = self.vae.loss(adj_hat, adj_pad, mu, logvar, beta=self.beta)
        return {"loss": losses["total"], "recon": losses["recon"], "kl": losses["kl"]}


@register("graph_diffusion", kind="model")
class GraphDiffusionModel(BaseModel):
    """Discrete graph denoising diffusion under the framework protocol."""

    def __init__(
        self,
        max_nodes: int = 20,
        hidden_dim: int = 128,
        timesteps: int = 100,
        *,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(
            config={
                "max_nodes": max_nodes,
                "hidden_dim": hidden_dim,
                "timesteps": timesteps,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "lr": lr,
            }
        )
        self.diffusion = DenoisingDiffusion(
            max_nodes=max_nodes,
            hidden_dim=hidden_dim,
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.max_nodes = max_nodes

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        features, adj = batch
        adj_pad, feat_pad = _pad(adj, features, self.max_nodes)
        loss = self.diffusion(feat_pad, adj_pad)
        return {"loss": loss}
