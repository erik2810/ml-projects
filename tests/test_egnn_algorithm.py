"""Framework wiring for the EGNN denoiser.

Verifies the model and dataset register under the expected names and that a
short training run reduces the denoising loss. The equivariance itself is
covered separately by ``test_equivariance.py``.
"""

from __future__ import annotations

from algorithms.physics_gnn import EGNNDenoiseModel, NoisyMeshDataset
from core import Trainer, list_registered
from core.training.trainer import seed_everything


def test_egnn_components_registered() -> None:
    models = set(list_registered("model")["model"])
    datasets = set(list_registered("dataset")["dataset"])
    assert "egnn_denoise" in models
    assert "noisy_mesh" in datasets


def test_egnn_denoise_reduces_loss() -> None:
    seed_everything(0)
    ds = NoisyMeshDataset(rows=6, cols=6, noise=0.15, seed=0)
    model = EGNNDenoiseModel(in_channels=ds.h.size(1), hidden_channels=32, num_layers=4, lr=1e-2)

    result = Trainer(model, max_epochs=5, seed=0, device="cpu").fit(ds)
    history = result.history["train_loss"]

    assert len(history) == 5
    # Denoising pulls the noised mesh back toward the clean one, so the MSE drops.
    assert history[-1] < history[0]
