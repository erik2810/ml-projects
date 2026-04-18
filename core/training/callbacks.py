"""Trainer callbacks.

Callbacks receive the Trainer instance on each hook and may read its public
attributes (``model``, ``metrics``, ``epoch``, ``should_stop``). They never
modify the training loop directly; to stop early they set ``trainer.should_stop``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.training.trainer import Trainer


class Callback:
    """No-op base. Subclasses override hooks of interest."""

    def on_train_start(self, trainer: "Trainer") -> None: ...
    def on_train_end(self, trainer: "Trainer") -> None: ...
    def on_epoch_start(self, trainer: "Trainer") -> None: ...
    def on_epoch_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None: ...


class EarlyStopping(Callback):
    """Stop when a monitored metric fails to improve for ``patience`` epochs."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self._best: float | None = None
        self._bad_epochs = 0

    def on_epoch_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor]
        improved = (
            self._best is None
            or (self.mode == "min" and current < self._best - self.min_delta)
            or (self.mode == "max" and current > self._best + self.min_delta)
        )
        if improved:
            self._best = current
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
            if self._bad_epochs >= self.patience:
                trainer.should_stop = True


class Checkpoint(Callback):
    """Save model weights when the monitored metric improves.

    Only writes on improvement by default (``save_best_only=True``). The
    filename is formatted with the current epoch and monitored value.
    """

    def __init__(
        self,
        dirpath: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        filename: str = "model-{epoch:03d}-{metric:.4f}.pt",
        save_best_only: bool = True,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.filename = filename
        self.save_best_only = save_best_only
        self._best: float | None = None
        self.last_path: Path | None = None

    def on_epoch_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        current = metrics.get(self.monitor)
        if current is None:
            return
        improved = (
            self._best is None
            or (self.mode == "min" and current < self._best)
            or (self.mode == "max" and current > self._best)
        )
        if self.save_best_only and not improved:
            return

        self.dirpath.mkdir(parents=True, exist_ok=True)
        path = self.dirpath / self.filename.format(epoch=trainer.epoch, metric=current)
        save_fn = getattr(trainer.model, "save", None)
        if callable(save_fn):
            save_fn(path)
        else:
            import torch

            torch.save({"state_dict": trainer.model.state_dict()}, path)
        self.last_path = path
        if improved:
            self._best = current


class ProgressLogger(Callback):
    """Print compact per-epoch progress. Every N epochs by default."""

    def __init__(self, every: int = 1, keys: list[str] | None = None) -> None:
        self.every = max(1, every)
        self.keys = keys

    def on_epoch_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        if trainer.epoch % self.every != 0 and trainer.epoch != trainer.max_epochs:
            return
        keys = self.keys or list(metrics.keys())
        parts = [f"epoch {trainer.epoch:>3d}/{trainer.max_epochs}"]
        for k in keys:
            if k in metrics:
                parts.append(f"{k}={metrics[k]:.4f}")
        print(" | ".join(parts))


def _format_metrics(metrics: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
