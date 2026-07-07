"""Framework adapter for the E(n)-equivariant GNN.

Registers an equivariant coordinate-denoising task, which is the natural way to
exercise EGNN in the framework. The ``spring_mesh`` node-classification task
labels nodes by their absolute x-position, a quantity no rotation-invariant model
can recover, so it is the wrong benchmark for a strict EGNN. Denoising is right:
the model sees noised positions and predicts the clean ones, and because EGNN's
coordinate output is SE(3)-equivariant, the MSE to the clean target is invariant
under a joint rigid motion of input and target. The task respects the symmetry
the model is built around.

Batch protocol::

    (h, noisy_positions, adj, clean_positions, mask)

where ``h`` is a per-node invariant feature (node degree).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from algorithms.physics_gnn.dataset import _grid_mesh
from backend.core.physics_gnn.models import EGNN
from core.datasets.base import BaseDataset
from core.models.base import BaseModel
from core.registry import register


@register("noisy_mesh", kind="dataset")
class NoisyMeshDataset(BaseDataset):
    """A 3D mesh with Gaussian-noised node positions; the target is the clean mesh."""

    def __init__(
        self,
        rows: int = 6,
        cols: int = 6,
        noise: float = 0.15,
        seed: int = 0,
    ) -> None:
        super().__init__(config={"rows": rows, "cols": cols, "noise": noise, "seed": seed})
        clean, adj = _grid_mesh(rows, cols, seed=seed)

        g = torch.Generator().manual_seed(seed + 1)
        noisy = clean + noise * torch.randn(clean.shape, generator=g)

        self.h = adj.sum(dim=1, keepdim=True)  # (N, 1) node degree, an invariant feature
        self.noisy = noisy
        self.adj = adj
        self.clean = clean
        self.mask = torch.ones(clean.size(0), dtype=torch.bool)

    def _batch(self) -> tuple[torch.Tensor, ...]:
        return self.h, self.noisy, self.adj, self.clean, self.mask

    def train_batches(self) -> list[tuple[torch.Tensor, ...]]:
        return [self._batch()]

    def val_batches(self) -> list[tuple[torch.Tensor, ...]]:
        return [self._batch()]


@register("egnn_denoise", kind="model")
class EGNNDenoiseModel(BaseModel):
    """Equivariant coordinate denoiser built on :class:`EGNN`."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 4,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(
            config={
                "in_channels": in_channels,
                "hidden_channels": hidden_channels,
                "num_layers": num_layers,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
        self.net = EGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,  # invariant head unused by the denoising task
            num_layers=num_layers,
            update_coordinates=True,
        )

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        h, noisy, adj, clean, mask = batch
        _, pred = self.net(h, noisy, adj)
        loss = F.mse_loss(pred[mask], clean[mask])
        with torch.no_grad():
            rmse = loss.sqrt()
        return {"loss": loss, "rmse": rmse}
