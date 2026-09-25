"""Lightweight metric tracker.

Stores scalar metrics per epoch in a dict-of-lists layout so they serialize
cleanly to JSON and render directly in the frontend's loss curves.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


class MetricTracker:
    """Append-only container for per-epoch scalar metrics."""

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = defaultdict(list)

    def log(self, metrics: dict[str, Any], prefix: str = "") -> None:
        """Append all scalar entries in ``metrics`` under optional key prefix."""
        for key, value in metrics.items():
            try:
                self._history[f"{prefix}{key}"].append(_to_float(value))
            except (TypeError, ValueError):
                continue

    def history(self) -> dict[str, list[float]]:
        """Full metric history as a JSON-serializable dict."""
        return {k: list(v) for k, v in self._history.items()}

    def latest(self) -> dict[str, float]:
        """Most recent value of every tracked metric."""
        return {k: v[-1] for k, v in self._history.items() if v}

    def best(self, key: str, mode: str = "min") -> float | None:
        """Return min/max of ``key`` across its history, or None if unseen."""
        values = self._history.get(key)
        if not values:
            return None
        return min(values) if mode == "min" else max(values)

    def __contains__(self, key: str) -> bool:
        return key in self._history

    def __getitem__(self, key: str) -> list[float]:
        return list(self._history[key])
