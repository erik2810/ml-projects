"""Pydantic schema for experiment configurations.

An experiment is the tuple (model, dataset, training). Each sub-config names
a registered component (``name``) plus a free-form ``params`` dict forwarded
to that component's constructor. Keeping params loose rather than fully typed
lets algorithm authors add hyperparameters without touching the schema.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Which model class to instantiate and with what hyperparameters."""

    name: str = Field(..., description="Registered model name (see core.registry).")
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    """Which dataset class to instantiate and with what hyperparameters."""

    name: str = Field(..., description="Registered dataset name (see core.registry).")
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Trainer-level hyperparameters shared across algorithms."""

    max_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = None
    device: str | None = Field(
        default=None,
        description="'cpu', 'cuda', or None to auto-detect.",
    )
    seed: int | None = 42
    early_stopping: dict[str, Any] | None = None
    checkpoint_dir: str | None = None
    log_every: int = 10


class ExperimentConfig(BaseModel):
    """Top-level experiment descriptor. Load/dump via ``core.config.loader``."""

    name: str
    description: str = ""
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tags: list[str] = Field(default_factory=list)
