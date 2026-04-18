"""Framework-native GNN models.

Thin adapters around :class:`backend.core.gnn.model.NodeClassifier` and
:class:`~backend.core.gnn.model.GraphClassifier`. They inherit from
:class:`core.BaseModel` so :class:`core.Trainer` can drive them via the
standard ``training_step`` / ``validation_step`` protocol.

The batch shape for node classification is a 4-tuple:

    (features, adj, labels, train_mask)

For graph classification it is a 3-tuple:

    (features_list, adj_list, labels)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from backend.core.gnn.model import GraphClassifier, NodeClassifier
from core.models.base import BaseModel
from core.registry import register


# --------------------------------------------------------------------- #
# Node classification                                                   #
# --------------------------------------------------------------------- #


class _NodeClassifierBase(BaseModel):
    """Shared logic for GCN/GAT node classifiers under the framework."""

    _layer_type: str = "gcn"

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        hidden: int = 16,
        n_layers: int = 2,
        dropout: float = 0.5,
        n_heads: int = 4,
        lr: float = 1e-2,
        weight_decay: float = 5e-4,
    ) -> None:
        super().__init__(
            config={
                "in_features": in_features,
                "num_classes": num_classes,
                "hidden": hidden,
                "n_layers": n_layers,
                "dropout": dropout,
                "n_heads": n_heads,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
        self.net = NodeClassifier(
            in_features=in_features,
            hidden=hidden,
            num_classes=num_classes,
            n_layers=n_layers,
            dropout=dropout,
            layer_type=self._layer_type,
            n_heads=n_heads,
        )

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.net(features, adj)

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        features, adj, labels, train_mask = batch
        logits = self.net(features, adj)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        with torch.no_grad():
            preds = logits[train_mask].argmax(dim=1)
            acc = (preds == labels[train_mask]).float().mean()
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        features, adj, labels, train_mask = batch
        logits = self.net(features, adj)
        val_mask = ~train_mask
        if val_mask.sum() == 0:  # no held-out nodes available
            val_mask = train_mask
        loss = F.cross_entropy(logits[val_mask], labels[val_mask])
        preds = logits[val_mask].argmax(dim=1)
        acc = (preds == labels[val_mask]).float().mean()
        return {"loss": loss, "acc": acc}


@register("gcn_node", kind="model")
class GCNNodeClassifier(_NodeClassifierBase):
    """Framework-native GCN node classifier."""

    _layer_type = "gcn"


@register("gat_node", kind="model")
class GATNodeClassifier(_NodeClassifierBase):
    """Framework-native GAT node classifier."""

    _layer_type = "gat"


# --------------------------------------------------------------------- #
# Graph classification                                                  #
# --------------------------------------------------------------------- #


@register("gcn_graph", kind="model")
class GCNGraphClassifier(BaseModel):
    """GCN encoder + global pooling + MLP head for graph-level prediction."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        hidden: int = 32,
        n_layers: int = 3,
        dropout: float = 0.5,
        pool: str = "mean",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            config={
                "in_features": in_features,
                "num_classes": num_classes,
                "hidden": hidden,
                "n_layers": n_layers,
                "dropout": dropout,
                "pool": pool,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
        self.net = GraphClassifier(
            in_features=in_features,
            hidden=hidden,
            num_classes=num_classes,
            n_layers=n_layers,
            dropout=dropout,
            layer_type="gcn",
            pool=pool,
        )

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        features_list, adj_list, labels = batch
        logits = self.net(features_list, adj_list)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()
        return {"loss": loss, "acc": acc}
