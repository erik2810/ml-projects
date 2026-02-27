import torch
import pytest

from backend.core.graphvae.model import (
    GraphVAE, DenoisingDiffusion, GraphEncoder, GraphDecoder,
    generate_social_skeletons,
)
from backend.core.graph_utils import erdos_renyi, one_hot_degree_features


def test_encoder_output_shape():
    enc = GraphEncoder(in_dim=11, hidden_dim=16, latent_dim=8)
    adj = erdos_renyi(10, 0.3)
    x = one_hot_degree_features(adj, max_degree=10)
    mu, logvar = enc(x, adj)
    assert mu.shape == (8,)
    assert logvar.shape == (8,)


def test_decoder_output_shape():
    dec = GraphDecoder(latent_dim=8, hidden_dim=32, max_nodes=10, node_feat_dim=11)
    z = torch.randn(8)
    adj_hat, feat_hat = dec(z)
    assert adj_hat.shape == (10, 10)
    assert feat_hat.shape == (10, 11)
    # adj should be in [0, 1] (sigmoid output)
    assert adj_hat.min() >= 0.0 and adj_hat.max() <= 1.0


def test_vae_forward():
    model = GraphVAE(max_nodes=10, node_feat_dim=11, latent_dim=8, hidden_dim=16)
    adj = erdos_renyi(10, 0.3)
    x = one_hot_degree_features(adj, max_degree=10)
    adj_hat, feat_hat, mu, logvar = model(x, adj)
    assert adj_hat.shape == (10, 10)
    assert mu.shape == (8,)


def test_vae_loss():
    model = GraphVAE(max_nodes=10, node_feat_dim=11, latent_dim=8, hidden_dim=16)
    adj = erdos_renyi(10, 0.3)
    x = one_hot_degree_features(adj, max_degree=10)
    adj_hat, _, mu, logvar = model(x, adj)
    loss_dict = model.loss(adj_hat, adj, mu, logvar)
    assert "total" in loss_dict
    assert loss_dict["total"].dim() == 0
    assert loss_dict["recon"].item() >= 0


def test_generate_social_skeletons():
    graphs = generate_social_skeletons(10, [6, 8])
    assert len(graphs) == 10
    for feat, adj in graphs:
        assert adj.shape[0] == adj.shape[1]
        assert feat.shape[0] == adj.shape[0]
        # symmetric
        assert torch.allclose(adj, adj.t())


def test_diffusion_forward():
    adj = erdos_renyi(8, 0.3)
    x = one_hot_degree_features(adj, max_degree=10)
    model = DenoisingDiffusion(max_nodes=8, hidden_dim=16, timesteps=10)
    loss = model(x, adj)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_diffusion_sample():
    model = DenoisingDiffusion(max_nodes=6, hidden_dim=16, timesteps=5)
    adj = model.sample()
    assert adj.shape == (6, 6)
    # should be symmetric
    assert torch.allclose(adj, adj.t())
    # binary
    assert set(adj.unique().tolist()).issubset({0.0, 1.0})
