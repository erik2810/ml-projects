"""Smoke tests for the Phase-3 algorithm migrations.

Each migrated module gets light coverage: verify registration, dataset shape,
and a short :class:`core.Trainer` run. The goal is to prove the wrappers plug
into the framework cleanly — numeric correctness is already covered by the
legacy tests under ``test_graphvae.py`` etc. against the wrapped classes.
"""

from __future__ import annotations

import torch

from algorithms.generator import ConditionalGenerator, ConditionalGraphsDataset
from algorithms.graphvae import (
    GraphVAEModel,
    GraphDiffusionModel,
    SocialSkeletonsDataset,
)
from algorithms.hyperbolic import (
    HyperbolicEmbeddingModel,
    HyperbolicGNNModel,
    TreeGraphDataset,
)
from algorithms.physics_gnn import PhysicsGNNNodeModel, SpringMeshDataset
from core import Trainer, list_registered
from core.training.trainer import seed_everything


# --------------------------------------------------------------------- #
# Registry                                                              #
# --------------------------------------------------------------------- #


def test_all_migrations_registered():
    models = set(list_registered("model")["model"])
    datasets = set(list_registered("dataset")["dataset"])
    for name in (
        "graph_vae",
        "graph_diffusion",
        "cond_graph_vae",
        "hyperbolic_gnn",
        "hyperbolic_embedding",
        "physics_gnn_node",
    ):
        assert name in models, f"missing model: {name}"
    for name in (
        "social_skeletons",
        "random_graph_bank",
        "tree_graph",
        "spring_mesh",
    ):
        assert name in datasets, f"missing dataset: {name}"


# --------------------------------------------------------------------- #
# graphvae                                                              #
# --------------------------------------------------------------------- #


def test_graph_vae_trains_on_social_skeletons():
    seed_everything(0)
    ds = SocialSkeletonsDataset(num_graphs=6, sizes=[8, 10], seed=0)
    model = GraphVAEModel(
        max_nodes=12,
        node_feat_dim=ds.train_batches()[0][0].size(1),
        latent_dim=16,
        hidden_dim=32,
        beta=0.1,
    )
    result = Trainer(model, max_epochs=3, seed=0, device="cpu").fit(ds)
    assert "train_loss" in result.history
    assert "train_recon" in result.history
    assert "train_kl" in result.history
    assert len(result.history["train_loss"]) == 3


def test_graph_diffusion_trains_step():
    seed_everything(0)
    ds = SocialSkeletonsDataset(num_graphs=3, sizes=[8, 10], seed=0)
    model = GraphDiffusionModel(max_nodes=12, hidden_dim=32, timesteps=10)
    result = Trainer(model, max_epochs=2, seed=0, device="cpu").fit(ds)
    assert len(result.history["train_loss"]) == 2


# --------------------------------------------------------------------- #
# generator                                                             #
# --------------------------------------------------------------------- #


def test_conditional_generator_trains():
    seed_everything(0)
    ds = ConditionalGraphsDataset(num_graphs=32, max_nodes=8, batch_size=16, seed=0)
    model = ConditionalGenerator(max_nodes=8, latent_dim=8, hidden_dim=32)
    result = Trainer(model, max_epochs=2, seed=0, device="cpu").fit(ds)
    history = result.history
    assert "train_loss" in history and "train_recon" in history and "train_kl" in history
    assert history["train_loss"][-1] < history["train_loss"][0] + 1.0  # at least not diverging


# --------------------------------------------------------------------- #
# hyperbolic                                                            #
# --------------------------------------------------------------------- #


def test_hyperbolic_gnn_trains_on_tree():
    seed_everything(0)
    ds = TreeGraphDataset(depth=2, branching=2, num_classes=2, per_class=2, seed=0)
    model = HyperbolicGNNModel(
        in_channels=ds.features.size(1),
        out_channels=2,
        hidden_channels=8,
        num_layers=2,
        dropout=0.0,
        lr=1e-2,
    )
    result = Trainer(model, max_epochs=10, seed=0, device="cpu").fit(ds)
    assert result.history["train_loss"][-1] < result.history["train_loss"][0] + 0.5


def test_hyperbolic_embedding_trains():
    seed_everything(0)
    pos_edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long)
    neg_edges = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)

    class EdgeBatches:
        def train_batches(self):
            return [(pos_edges, neg_edges)]

    model = HyperbolicEmbeddingModel(num_nodes=4, embed_dim=4, lr=5e-2)
    result = Trainer(model, max_epochs=5, seed=0, device="cpu").fit(EdgeBatches())
    assert len(result.history["train_loss"]) == 5


# --------------------------------------------------------------------- #
# physics_gnn                                                           #
# --------------------------------------------------------------------- #


def test_physics_gnn_node_trains():
    seed_everything(0)
    ds = SpringMeshDataset(rows=4, cols=4, per_class=2, seed=0)
    model = PhysicsGNNNodeModel(
        out_channels=2,
        in_channels=0,
        hidden_channels=8,
        num_layers=2,
        use_diffusion=False,
        use_reaction_diffusion=False,
        use_attention=False,
        regularise=False,
        dropout=0.0,
        physics_weight=0.0,
        lr=1e-2,
    )
    result = Trainer(model, max_epochs=3, seed=0, device="cpu").fit(ds)
    assert "train_loss" in result.history
    assert len(result.history["train_loss"]) == 3
