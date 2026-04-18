"""Tests for the core framework abstractions.

Uses a deliberately trivial model (linear regression) so the tests exercise
the Trainer / callbacks / metrics machinery without depending on any of the
algorithm modules under ``backend/core``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from core import (
    BaseModel,
    Checkpoint,
    EarlyStopping,
    ExperimentConfig,
    MetricTracker,
    ProgressLogger,
    Trainer,
    dump_config,
    get,
    list_registered,
    load_config,
    register,
)


# --------------------------------------------------------------------- #
# Test fixtures: a trivial model + dataset                              #
# --------------------------------------------------------------------- #


class TinyRegressor(BaseModel):
    def __init__(self, in_dim: int = 4, lr: float = 1e-2) -> None:
        super().__init__(config={"in_dim": in_dim, "lr": lr})
        self.linear = nn.Linear(in_dim, 1)

    def training_step(self, batch):
        x, y = batch
        pred = self.linear(x).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y)
        return {"loss": loss}


class _XYDataset:
    def __init__(self, n: int = 64, in_dim: int = 4, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(in_dim, generator=g)
        x = torch.randn(n, in_dim, generator=g)
        y = x @ w + 0.01 * torch.randn(n, generator=g)
        self._train = [(x[:48], y[:48])]
        self._val = [(x[48:], y[48:])]

    def train_batches(self):
        return self._train

    def val_batches(self):
        return self._val


# --------------------------------------------------------------------- #
# BaseModel                                                              #
# --------------------------------------------------------------------- #


def test_base_model_configure_optimizers_uses_config_lr():
    model = TinyRegressor(lr=0.05)
    opt = model.configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)


def test_base_model_save_and_load_roundtrip(tmp_path: Path):
    model = TinyRegressor(in_dim=4, lr=0.01)
    with torch.no_grad():
        model.linear.weight.fill_(0.5)
    ckpt = tmp_path / "model.pt"
    model.save(ckpt)

    restored = TinyRegressor.load(ckpt)
    assert torch.allclose(restored.linear.weight, model.linear.weight)
    assert restored.get_config() == {"in_dim": 4, "lr": 0.01}


# --------------------------------------------------------------------- #
# Trainer                                                                #
# --------------------------------------------------------------------- #


def test_trainer_fit_reduces_loss():
    model = TinyRegressor()
    trainer = Trainer(model, max_epochs=30, seed=0, device="cpu")
    result = trainer.fit(_XYDataset())
    history = result.history
    assert "train_loss" in history
    # Loss should actually go down on a linear problem.
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert result.epochs_run == 30
    assert not result.stopped_early


def test_trainer_runs_validation_when_dataset_provides_it():
    model = TinyRegressor()
    trainer = Trainer(model, max_epochs=3, seed=0, device="cpu")
    result = trainer.fit(_XYDataset())
    assert "val_loss" in result.history
    assert len(result.history["val_loss"]) == 3


def test_trainer_respects_should_stop_via_callback():
    class StopAfterTwo:
        def on_epoch_end(self, trainer, metrics):
            if trainer.epoch >= 2:
                trainer.should_stop = True

    model = TinyRegressor()
    trainer = Trainer(model, max_epochs=100, seed=0, device="cpu", callbacks=[StopAfterTwo()])
    result = trainer.fit(_XYDataset())
    assert result.epochs_run == 2
    assert result.stopped_early


def test_trainer_raises_when_training_step_missing_loss():
    class BadModel(BaseModel):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(4, 1)

        def training_step(self, batch):
            return {"not_loss": torch.tensor(1.0)}

    with pytest.raises(KeyError, match="must return a dict containing 'loss'"):
        Trainer(BadModel(), max_epochs=1, device="cpu").fit(_XYDataset())


# --------------------------------------------------------------------- #
# Callbacks                                                              #
# --------------------------------------------------------------------- #


def test_early_stopping_triggers_on_plateau():
    # Monotone-increasing val_loss → patience exhausted → stop
    class FlatModel(BaseModel):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
            self._step = 0

        def training_step(self, batch):
            self._step += 1
            return {"loss": self.param.sum() * 0 + float(self._step)}

    trainer = Trainer(
        FlatModel(),
        max_epochs=20,
        callbacks=[EarlyStopping(monitor="train_loss", patience=3)],
        device="cpu",
    )
    result = trainer.fit(_XYDataset())
    assert result.stopped_early
    assert result.epochs_run < 20


def test_checkpoint_saves_on_improvement(tmp_path: Path):
    cb = Checkpoint(dirpath=tmp_path, monitor="train_loss", mode="min", save_best_only=True)
    model = TinyRegressor()
    Trainer(model, max_epochs=5, seed=0, device="cpu", callbacks=[cb]).fit(_XYDataset())
    assert cb.last_path is not None and cb.last_path.exists()


def test_progress_logger_prints_without_error(capsys):
    model = TinyRegressor()
    Trainer(
        model,
        max_epochs=2,
        seed=0,
        device="cpu",
        callbacks=[ProgressLogger(every=1)],
    ).fit(_XYDataset())
    out = capsys.readouterr().out
    assert "epoch" in out
    assert "train_loss" in out


# --------------------------------------------------------------------- #
# MetricTracker                                                          #
# --------------------------------------------------------------------- #


def test_metric_tracker_history_and_best():
    tracker = MetricTracker()
    tracker.log({"loss": 0.9, "acc": 0.5})
    tracker.log({"loss": 0.7, "acc": 0.6})
    tracker.log({"loss": 0.75, "acc": 0.7})
    assert tracker["loss"] == [0.9, 0.7, 0.75]
    assert tracker.best("loss", mode="min") == 0.7
    assert tracker.best("acc", mode="max") == 0.7
    assert tracker.latest() == {"loss": 0.75, "acc": 0.7}


def test_metric_tracker_converts_tensors():
    tracker = MetricTracker()
    tracker.log({"loss": torch.tensor(0.5)}, prefix="train_")
    assert tracker["train_loss"] == [0.5]


# --------------------------------------------------------------------- #
# Registry                                                               #
# --------------------------------------------------------------------- #


def test_register_and_get_model():
    @register("tiny_reg_test", kind="model")
    class _Tiny(TinyRegressor):
        pass

    cls = get("tiny_reg_test", kind="model")
    assert cls is _Tiny
    assert "tiny_reg_test" in list_registered("model")["model"]


def test_register_duplicate_raises():
    @register("dup_test", kind="model")
    class _A(TinyRegressor):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register("dup_test", kind="model")
        class _B(TinyRegressor):
            pass


def test_get_unknown_raises_with_available_list():
    with pytest.raises(KeyError, match="Available model"):
        get("this_does_not_exist", kind="model")


# --------------------------------------------------------------------- #
# Config                                                                 #
# --------------------------------------------------------------------- #


def test_experiment_config_roundtrip_json(tmp_path: Path):
    cfg = ExperimentConfig(
        name="demo",
        description="unit test",
        model={"name": "tiny_reg", "params": {"in_dim": 4}},
        dataset={"name": "xy", "params": {}},
    )
    path = tmp_path / "cfg.json"
    dump_config(cfg, path)
    assert json.loads(path.read_text())["name"] == "demo"

    restored = load_config(path)
    assert restored.name == "demo"
    assert restored.model.name == "tiny_reg"
    assert restored.training.max_epochs == 100  # default


def test_experiment_config_yaml_roundtrip_if_available(tmp_path: Path):
    pytest.importorskip("yaml")
    cfg = ExperimentConfig(
        name="demo",
        model={"name": "tiny_reg"},
        dataset={"name": "xy"},
    )
    path = tmp_path / "cfg.yaml"
    dump_config(cfg, path)
    restored = load_config(path)
    assert restored.name == "demo"
