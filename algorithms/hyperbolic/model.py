"""Framework adapters for the hyperbolic models.

Hyperbolic parameters live on the Poincaré ball and require a Riemannian
optimizer; the remaining Euclidean weights use plain AdamW. The wrappers
implement :meth:`BaseModel.configure_optimizers` to return both, which
:class:`core.Trainer` already knows how to drive.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from backend.core.hyperbolic.manifolds import RiemannianAdam
from backend.core.hyperbolic.models import HyperbolicEmbedding, HyperbolicGNN
from core.models.base import BaseModel
from core.registry import register


def _split_params(model: torch.nn.Module):
    manifold, euclidean = [], []
    for p in model.parameters():
        if getattr(p, "manifold", None) is not None:
            manifold.append(p)
        else:
            euclidean.append(p)
    return manifold, euclidean


@register("hyperbolic_gnn", kind="model")
class HyperbolicGNNModel(BaseModel):
    """Node classifier on the Poincaré ball.

    Batch protocol: ``(features, adj, labels, train_mask)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 64,
        num_layers: int = 2,
        c: float = 1.0,
        dropout: float = 0.1,
        use_attention: bool = False,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            config={
                "in_channels": in_channels,
                "out_channels": out_channels,
                "hidden_channels": hidden_channels,
                "num_layers": num_layers,
                "c": c,
                "dropout": dropout,
                "use_attention": use_attention,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
        self.net = HyperbolicGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            c=c,
            dropout=dropout,
            use_attention=use_attention,
        )

    def configure_optimizers(self):
        manifold, euclidean = _split_params(self.net)
        lr = float(self._config["lr"])
        wd = float(self._config["weight_decay"])
        opts = []
        if manifold:
            opts.append(RiemannianAdam(manifold, lr=lr))
        if euclidean:
            opts.append(AdamW(euclidean, lr=lr, weight_decay=wd))
        return opts if len(opts) > 1 else opts[0]

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        features, adj, labels, train_mask = batch
        logits = self.net(features, adj)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        with torch.no_grad():
            acc = (logits[train_mask].argmax(dim=1) == labels[train_mask]).float().mean()
        return {"loss": loss, "acc": acc}


@register("hyperbolic_embedding", kind="model")
class HyperbolicEmbeddingModel(BaseModel):
    """Poincaré embeddings for link prediction.

    Batch protocol: ``(pos_edges, neg_edges)`` — both ``(E, 2)`` long tensors.
    """

    def __init__(
        self,
        num_nodes: int,
        *,
        embed_dim: int = 16,
        c: float = 1.0,
        t: float = 1.0,
        r: float = 2.0,
        init_scale: float = 0.01,
        lr: float = 1e-2,
    ) -> None:
        super().__init__(
            config={
                "num_nodes": num_nodes,
                "embed_dim": embed_dim,
                "c": c,
                "t": t,
                "r": r,
                "init_scale": init_scale,
                "lr": lr,
            }
        )
        self.net = HyperbolicEmbedding(
            num_nodes=num_nodes,
            embed_dim=embed_dim,
            c=c,
            t=t,
            r=r,
            init_scale=init_scale,
        )

    def configure_optimizers(self):
        manifold, euclidean = _split_params(self.net)
        lr = float(self._config["lr"])
        opts = []
        if manifold:
            opts.append(RiemannianAdam(manifold, lr=lr))
        if euclidean:
            opts.append(AdamW(euclidean, lr=lr))
        return opts if len(opts) > 1 else opts[0]

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        pos_edges, neg_edges = batch
        loss = self.net.loss(pos_edges, neg_edges)
        return {"loss": loss}
