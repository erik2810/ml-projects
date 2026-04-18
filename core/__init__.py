"""Core ML research framework.

Lightweight abstractions for reproducible experimentation:

- :class:`~core.models.BaseModel`: declarative model interface with training_step hooks.
- :class:`~core.training.Trainer`: unified training loop with callbacks.
- :class:`~core.config.ExperimentConfig`: YAML-backed experiment specification.
- :mod:`~core.registry`: string-keyed component lookup.

Existing module code under ``backend/core/*`` continues to work unchanged.
Algorithms are migrated onto this framework one at a time.
"""

from core.models.base import BaseModel
from core.training.trainer import Trainer
from core.training.callbacks import Callback, EarlyStopping, Checkpoint, ProgressLogger
from core.training.metrics import MetricTracker
from core.config.schema import ExperimentConfig, ModelConfig, TrainingConfig, DatasetConfig
from core.config.loader import load_config, dump_config
from core.registry import register, get, list_registered

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
