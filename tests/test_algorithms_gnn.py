"""Tests for the framework-native GNN reference implementation.

These verify that the ``algorithms.gnn`` wrappers plug cleanly into the core
framework: models reach similar accuracy to the legacy
``train_node_classifier`` loop, the dataset matches the full-batch protocol,
and registry lookups resolve correctly.
"""

from __future__ import annotations

import torch

from algorithms.gnn import (
    GCNGraphClassifier,
    GCNNodeClassifier,
    GATNodeClassifier,
    KarateClubDataset,
)
from core import Trainer, get, list_registered
from core.training.callbacks import EarlyStopping
from core.training.trainer import seed_everything


# --------------------------------------------------------------------- #
# Registry integration                                                  #
# --------------------------------------------------------------------- #


def test_gnn_components_are_registered():
    models = list_registered("model")["model"]
    datasets = list_registered("dataset")["dataset"]
    assert "gcn_node" in models
    assert "gat_node" in models
    assert "gcn_graph" in models
    assert "karate_club" in datasets


def test_lookup_by_name_returns_correct_classes():
    assert get("gcn_node", kind="model") is GCNNodeClassifier
    assert get("gat_node", kind="model") is GATNodeClassifier
    assert get("karate_club", kind="dataset") is KarateClubDataset


# --------------------------------------------------------------------- #
# Dataset                                                               #
# --------------------------------------------------------------------- #


def test_karate_club_dataset_batch_shapes():
    ds = KarateClubDataset(per_class=4, seed=0)
    (features, adj, labels, mask) = next(iter(ds.train_batches()))
    assert adj.shape == (34, 34)
    assert features.size(0) == 34
    assert labels.shape == (34,)
    assert mask.dtype == torch.bool
    # 4 per class on a 2-class problem = 8 training nodes
    assert mask.sum().item() == 8


def test_karate_club_train_mask_is_reproducible():
    a = KarateClubDataset(per_class=4, seed=123).train_mask
    b = KarateClubDataset(per_class=4, seed=123).train_mask
    c = KarateClubDataset(per_class=4, seed=999).train_mask
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# --------------------------------------------------------------------- #
# Training                                                              #
# --------------------------------------------------------------------- #


def test_gcn_node_classifier_fits_karate_club():
    seed_everything(0)
    ds = KarateClubDataset(per_class=4, seed=0)
    model = GCNNodeClassifier(
        in_features=ds.features.size(1),
        num_classes=2,
        hidden=16,
        n_layers=2,
        dropout=0.0,
        lr=1e-2,
    )
    result = Trainer(model, max_epochs=80, seed=0, device="cpu").fit(ds)
    history = result.history
    assert "train_loss" in history and "train_acc" in history
    assert "val_loss" in history and "val_acc" in history
    # Loss should descend and supervised accuracy should saturate.
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["train_acc"][-1] >= 0.9
    # The karate club is easy; we expect the model to generalise to the held-out nodes.
    assert history["val_acc"][-1] >= 0.8


def test_gat_node_classifier_trains():
    seed_everything(0)
    ds = KarateClubDataset(per_class=4, seed=0)
    model = GATNodeClassifier(
        in_features=ds.features.size(1),
        num_classes=2,
        hidden=8,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
        lr=1e-2,
    )
    result = Trainer(model, max_epochs=40, seed=0, device="cpu").fit(ds)
    # GAT can be noisier with few samples, but training loss must still drop.
    assert result.history["train_loss"][-1] < result.history["train_loss"][0]


def test_early_stopping_on_val_loss_applies():
    seed_everything(0)
    ds = KarateClubDataset(per_class=4, seed=0)
    model = GCNNodeClassifier(
        in_features=ds.features.size(1),
        num_classes=2,
        hidden=16,
        n_layers=2,
        dropout=0.0,
        lr=1e-2,
    )
    cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    result = Trainer(model, max_epochs=200, seed=0, device="cpu", callbacks=[cb]).fit(ds)
    # Karate club converges fast — patience must have fired well before 200 epochs.
    assert result.epochs_run < 200
    assert result.stopped_early


# --------------------------------------------------------------------- #
# Graph classifier                                                      #
# --------------------------------------------------------------------- #


def test_gcn_graph_classifier_forward_and_loss():
    seed_everything(0)
    from backend.core.graph_utils import erdos_renyi

    adjs = [erdos_renyi(n, 0.3) for n in (6, 8, 10, 12)]
    feats = [torch.randn(a.size(0), 4) for a in adjs]
    labels = torch.tensor([0, 1, 0, 1])

    model = GCNGraphClassifier(in_features=4, num_classes=2, hidden=8, n_layers=2, dropout=0.0)
    out = model.training_step((feats, adjs, labels))
    assert "loss" in out and "acc" in out
    assert out["loss"].requires_grad
    assert out["loss"].item() > 0
