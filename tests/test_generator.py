import torch
import pytest

from backend.core.generator.model import (
    ConditionalGraphGenerator, build_training_set, sample_graphs,
    _triu_size, _adj_to_triu_vec, _triu_vec_to_adj,
)


def test_triu_size():
    assert _triu_size(4) == 6
    assert _triu_size(10) == 45
    assert _triu_size(20) == 190


def test_adj_to_triu_roundtrip():
    adj = torch.zeros(5, 5)
    adj[0, 1] = adj[1, 0] = 1.0
    adj[2, 3] = adj[3, 2] = 1.0

    vec = _adj_to_triu_vec(adj)
    assert vec.shape == (_triu_size(5),)

    recovered = _triu_vec_to_adj(vec, 5)
    assert torch.allclose(recovered, adj)


def test_triu_batch():
    batch = torch.zeros(3, 5, 5)
    batch[0, 0, 1] = batch[0, 1, 0] = 1.0
    vecs = _adj_to_triu_vec(batch)
    assert vecs.shape == (3, _triu_size(5))

    recovered = _triu_vec_to_adj(vecs, 5)
    assert torch.allclose(recovered, batch)


def test_model_forward():
    model = ConditionalGraphGenerator(max_nodes=10, latent_dim=8, hidden_dim=32)
    vec = torch.randn(2, _triu_size(10))
    cond = torch.rand(2, 3)
    recon, mu, logvar = model(vec, cond)
    assert recon.shape == (2, _triu_size(10))
    assert mu.shape == (2, 8)


def test_model_loss():
    model = ConditionalGraphGenerator(max_nodes=10, latent_dim=8, hidden_dim=32)
    vec = torch.randint(0, 2, (_triu_size(10),)).float().unsqueeze(0)
    cond = torch.rand(1, 3)
    total, recon, kl = model.loss(vec, cond)
    assert total.dim() == 0  # scalar
    assert recon.item() >= 0
    assert kl.item() >= 0


def test_build_training_set():
    adj_vecs, conds = build_training_set(20, max_nodes=10)
    assert adj_vecs.shape == (20, _triu_size(10))
    assert conds.shape == (20, 3)
    # conditions should be normalized
    assert conds[:, 0].max() <= 1.0  # nodes_norm
    assert conds[:, 1].max() <= 1.0  # density


def test_sample_graphs():
    model = ConditionalGraphGenerator(max_nodes=8, latent_dim=4, hidden_dim=16)
    cond = torch.tensor([[0.5, 0.3, 0.2]])
    graphs = sample_graphs(model, cond, num_samples=3)
    assert graphs.shape == (3, 8, 8)
    # should be symmetric
    assert torch.allclose(graphs, graphs.transpose(1, 2))
    # should be binary
    assert set(graphs.unique().tolist()).issubset({0.0, 1.0})
