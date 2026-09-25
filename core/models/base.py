"""BaseModel: a minimal opinionated interface for framework-aware models.

Subclasses declare :meth:`training_step` (and optionally :meth:`validation_step`
and :meth:`configure_optimizers`). The :class:`~core.training.Trainer` drives
the outer loop, device placement, and callbacks without needing to know
anything about the model's internals.

A BaseModel remains a regular :class:`torch.nn.Module`, so it stays compatible
with existing utilities (``.parameters()``, ``.state_dict()``, ``torch.save``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class BaseModel(nn.Module):
    """Base class for models that plug into :class:`~core.training.Trainer`.

    Concrete subclasses must implement :meth:`training_step` and may override
    :meth:`validation_step`, :meth:`test_step`, and :meth:`configure_optimizers`.
    The return dict from each step must contain a ``"loss"`` scalar tensor
    during training; any additional entries are logged as metrics.
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self._config: dict = dict(config) if config else {}

    # ------------------------------------------------------------------ #
    # Hooks subclasses override                                          #
    # ------------------------------------------------------------------ #

    def training_step(self, batch: Any) -> dict[str, torch.Tensor]:
        """Compute loss (and optional metrics) for one batch.

        Must return a dict with a ``"loss"`` key whose value is a scalar
        tensor on which ``.backward()`` can be called. Additional entries
        are treated as metrics and forwarded to callbacks.
        """
        raise NotImplementedError

    def validation_step(self, batch: Any) -> dict[str, torch.Tensor]:
        """Compute validation metrics for one batch.

        Defaults to :meth:`training_step` under ``torch.no_grad``. Override
        when validation requires different outputs (e.g. classification
        accuracy, reconstruction error).
        """
        with torch.no_grad():
            return self.training_step(batch)

    def test_step(self, batch: Any) -> dict[str, torch.Tensor]:
        """Compute test metrics for one batch. Defaults to validation_step."""
        return self.validation_step(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer | list[torch.optim.Optimizer]:
        """Return the optimizer(s) used by the trainer.

        Default: Adam over all parameters with ``lr`` from config
        (falling back to 1e-3).
        """
        lr = float(self._config.get("lr", 1e-3))
        weight_decay = float(self._config.get("weight_decay", 0.0))
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------ #
    # Checkpoint helpers                                                 #
    # ------------------------------------------------------------------ #

    def get_config(self) -> dict:
        """Return a copy of the config used to build this model."""
        return dict(self._config)

    def save(self, path: str | Path) -> None:
        """Serialize weights + config to disk (single ``.pt`` bundle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self._config,
                "class": f"{type(self).__module__}.{type(self).__name__}",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "BaseModel":
        """Load a checkpoint previously produced by :meth:`save`.

        The stored config is forwarded to ``cls(**config)``. Subclasses whose
        constructor does not accept the full config dict should override this.
        """
        bundle = torch.load(Path(path), map_location=map_location, weights_only=False)
        model = cls(**bundle["config"]) if bundle["config"] else cls()
        model.load_state_dict(bundle["state_dict"])
        return model
