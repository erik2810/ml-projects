"""Framework adapter for PhysicsInformedGNN.

Batch protocol for node tasks::

    (positions, adj, x, labels, train_mask)

where ``x`` may be ``None`` (positions + curvature features are enough). The
model forwards (positions, adj, x, faces=None, return_energy=True) and adds
the physics regulariser to the task loss when ``regularise=True``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from backend.core.physics_gnn.models import PhysicsInformedGNN
from core.models.base import BaseModel
from core.registry import register


@register("physics_gnn_node", kind="model")
class PhysicsGNNNodeModel(BaseModel):
    """Physics-informed node classifier / regressor."""

    def __init__(
        self,
        out_channels: int,
        *,
        in_channels: int = 0,
        hidden_channels: int = 32,
        num_layers: int = 3,
        task: str = "node",
        use_curvature: bool = True,
        use_diffusion: bool = True,
        use_reaction_diffusion: bool = True,
        use_attention: bool = True,
        regularise: bool = True,
        dropout: float = 0.1,
        physics_weight: float = 0.01,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__(
            config={
                "in_channels": in_channels,
                "out_channels": out_channels,
                "hidden_channels": hidden_channels,
                "num_layers": num_layers,
                "task": task,
                "use_curvature": use_curvature,
                "use_diffusion": use_diffusion,
                "use_reaction_diffusion": use_reaction_diffusion,
                "use_attention": use_attention,
                "regularise": regularise,
                "dropout": dropout,
                "physics_weight": physics_weight,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
        self.net = PhysicsInformedGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            task=task,
            use_curvature=use_curvature,
            use_diffusion=use_diffusion,
            use_reaction_diffusion=use_reaction_diffusion,
            use_attention=use_attention,
            regularise=regularise,
            dropout=dropout,
        )
        self.physics_weight = physics_weight
        self.regularise = regularise

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        positions, adj, x, labels, train_mask = batch
        if self.regularise:
            logits, energy = self.net(positions, adj, x=x, return_energy=True)
        else:
            logits = self.net(positions, adj, x=x)
            energy = torch.zeros((), device=positions.device)

        task_loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss = task_loss + self.physics_weight * energy
        with torch.no_grad():
            preds = logits[train_mask].argmax(dim=1)
            acc = (preds == labels[train_mask]).float().mean()
        return {"loss": loss, "task_loss": task_loss, "physics": energy, "acc": acc}
