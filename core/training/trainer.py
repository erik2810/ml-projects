"""The unified training loop.

The :class:`Trainer` drives model training in a device- and callback-aware
way while delegating *what* to compute to the model's ``training_step`` and
``validation_step`` hooks.

Design notes:

- The dataset is iterated *every* epoch by calling ``dataset.train_batches()``
  (and ``val_batches()``). This matches the full-batch semi-supervised
  workloads that dominate this project while remaining compatible with
  mini-batch PyTorch ``DataLoader`` objects.
- Callbacks run in registration order. Ordering matters only when multiple
  callbacks both mutate ``trainer.should_stop``.
- Reproducibility: call :func:`seed_everything` before constructing the
  trainer if you need deterministic runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch

from core.training.callbacks import Callback
from core.training.metrics import MetricTracker


def seed_everything(seed: int) -> None:
    """Seed ``random``, numpy, and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainResult:
    """Outcome of a completed :meth:`Trainer.fit` call."""

    epochs_run: int
    stopped_early: bool
    history: dict[str, list[float]]
    best_metrics: dict[str, float] = field(default_factory=dict)


class Trainer:
    """Train a :class:`~core.models.BaseModel` (or compatible ``nn.Module``).

    Parameters
    ----------
    model:
        Either a :class:`~core.models.BaseModel` or any ``nn.Module`` that
        exposes ``training_step``.
    max_epochs:
        Upper bound on epochs. Callbacks may stop earlier via
        ``trainer.should_stop = True``.
    callbacks:
        Optional list of :class:`Callback` instances run on trainer hooks.
    device:
        ``"cpu"``, ``"cuda"``, or a :class:`torch.device`. Defaults to CUDA
        when available, otherwise CPU.
    grad_clip:
        If set, applies ``clip_grad_norm_`` with this max norm each step.
    seed:
        Optional seed; if given, :func:`seed_everything` is called on start.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_epochs: int = 100,
        *,
        callbacks: list[Callback] | None = None,
        device: str | torch.device | None = None,
        grad_clip: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.max_epochs = int(max_epochs)
        self.callbacks = list(callbacks or [])
        self.device = torch.device(device) if device else _default_device()
        self.grad_clip = grad_clip
        self.seed = seed

        self.metrics = MetricTracker()
        self.epoch: int = 0
        self.should_stop: bool = False
        self._optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer] | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(self, dataset: Any) -> TrainResult:
        """Run training against ``dataset`` for up to ``max_epochs`` epochs."""
        if self.seed is not None:
            seed_everything(self.seed)

        self.model.to(self.device)
        self._optimizer = self._resolve_optimizer()
        self._dispatch("on_train_start")

        stopped_early = False
        for epoch in range(1, self.max_epochs + 1):
            self.epoch = epoch
            self._dispatch("on_epoch_start")

            train_metrics = self._run_epoch(dataset, train=True)
            self.metrics.log(train_metrics, prefix="train_")

            if self._has_val(dataset):
                val_metrics = self._run_epoch(dataset, train=False)
                self.metrics.log(val_metrics, prefix="val_")

            epoch_metrics = self.metrics.latest()
            self._dispatch("on_epoch_end", epoch_metrics)

            if self.should_stop:
                stopped_early = True
                break

        self._dispatch("on_train_end")
        return TrainResult(
            epochs_run=self.epoch,
            stopped_early=stopped_early,
            history=self.metrics.history(),
            best_metrics=self._best_metrics(),
        )

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _run_epoch(self, dataset: Any, *, train: bool) -> dict[str, float]:
        if train:
            self.model.train()
            batches: Iterable[Any] = dataset.train_batches()
        else:
            self.model.eval()
            batches = dataset.val_batches() or []

        step_fn = self.model.training_step if train else getattr(
            self.model, "validation_step", self.model.training_step
        )

        accumulator: dict[str, list[float]] = {}
        for batch in batches:
            batch = self._move_to_device(batch)
            if train:
                outputs = self._training_step(step_fn, batch)
            else:
                with torch.no_grad():
                    outputs = step_fn(batch)
            for key, value in outputs.items():
                accumulator.setdefault(key, []).append(_scalar(value))

        return {k: float(np.mean(v)) if v else float("nan") for k, v in accumulator.items()}

    def _training_step(self, step_fn, batch: Any) -> dict[str, torch.Tensor]:
        outputs = step_fn(batch)
        if "loss" not in outputs:
            raise KeyError("training_step must return a dict containing 'loss'")
        loss = outputs["loss"]
        optimizers = self._optimizer if isinstance(self._optimizer, list) else [self._optimizer]
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        for opt in optimizers:
            opt.step()
        return outputs

    def _resolve_optimizer(self):
        if hasattr(self.model, "configure_optimizers"):
            return self.model.configure_optimizers()
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def _has_val(self, dataset: Any) -> bool:
        fn = getattr(dataset, "val_batches", None)
        if not callable(fn):
            return False
        sentinel = fn()
        return sentinel is not None and sentinel != []

    def _dispatch(self, hook: str, *args: Any) -> None:
        for cb in self.callbacks:
            method = getattr(cb, hook, None)
            if callable(method):
                method(self, *args)

    def _move_to_device(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(b) for b in batch)
        if isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        return batch

    def _best_metrics(self) -> dict[str, float]:
        best: dict[str, float] = {}
        for key in self.metrics.history():
            mode = "max" if any(t in key for t in ("acc", "accuracy", "f1", "auc")) else "min"
            value = self.metrics.best(key, mode=mode)
            if value is not None:
                best[key] = value
        return best


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)
