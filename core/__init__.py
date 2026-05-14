"""Core ML research framework.

Lightweight abstractions for reproducible experimentation:

- :class:`~core.models.BaseModel`: declarative model interface with training_step hooks.
- :class:`~core.training.Trainer`: unified training loop with callbacks.
- :class:`~core.config.ExperimentConfig`: YAML-backed experiment specification.
- :mod:`~core.registry`: string-keyed component lookup.

Existing module code under ``backend/core/*`` continues to work unchanged.
Algorithms are migrated onto this framework one at a time.
"""

from core.config.loader import dump_config, load_config
from core.config.schema import DatasetConfig, ExperimentConfig, ModelConfig, TrainingConfig
from core.models.base import BaseModel
from core.registry import get, list_registered, register
from core.training.callbacks import Callback, Checkpoint, EarlyStopping, ProgressLogger
from core.training.metrics import MetricTracker
from core.training.trainer import Trainer

__all__ = [
    "BaseModel",
    "Trainer",
    "Callback",
    "EarlyStopping",
    "Checkpoint",
    "ProgressLogger",
    "MetricTracker",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "DatasetConfig",
    "load_config",
    "dump_config",
    "register",
    "get",
    "list_registered",
]
