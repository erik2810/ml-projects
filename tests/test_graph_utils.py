import torch

from backend.core.graph_utils import (
    adj_to_edge_index, edge_index_to_adj, normalize_adj,
    clustering_coefficient, graph_density,
    erdos_renyi, barabasi_albert, watts_strogatz,
    one_hot_degree_features,
)


def test_adj_edge_index_roundtrip():
    adj = torch.tensor([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ], dtype=torch.float32)
    ei = adj_to_edge_index(adj)
    assert ei.shape[0] == 2
    recovered = edge_index_to_adj(ei, 3)
    assert torch.allclose(adj, recovered)


def test_normalize_adj_row_sums():
    adj = torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=torch.float32)
    a_norm = normalize_adj(adj, add_self_loops=True)
    # should be symmetric
    assert torch.allclose(a_norm, a_norm.t(), atol=1e-6)


def test_graph_density_complete():
    n = 5
    adj = torch.ones(n, n) - torch.eye(n)
    assert abs(graph_density(adj) - 1.0) < 1e-6


def test_graph_density_empty():
    adj = torch.zeros(5, 5)
    assert graph_density(adj) == 0.0


def test_clustering_triangle():
    # complete graph on 3 nodes: every node in a triangle
    adj = torch.tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=torch.float32)
    cc = clustering_coefficient(adj)
    assert cc.shape == (3,)
    # each node should have clustering coeff = 1
    assert torch.allclose(cc, torch.ones(3), atol=1e-5)


def test_erdos_renyi_properties():
    adj = erdos_renyi(20, 0.5)
    assert adj.shape == (20, 20)
    assert torch.allclose(adj, adj.t())
    assert adj.diagonal().sum() == 0  # no self-loops


def test_barabasi_albert_properties():
    adj = barabasi_albert(15, m=2)
    assert adj.shape == (15, 15)
    assert torch.allclose(adj, adj.t())
    assert adj.diagonal().sum() == 0


def test_watts_strogatz_properties():
    adj = watts_strogatz(12, k=4, p=0.1)
    assert adj.shape == (12, 12)
    assert torch.allclose(adj, adj.t())


def test_one_hot_degree_features():
    adj = torch.tensor([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ], dtype=torch.float32)
    feats = one_hot_degree_features(adj, max_degree=5)
    assert feats.shape == (3, 6)
    # node 0 has degree 2, nodes 1 and 2 have degree 1
    assert feats[0, 2] == 1.0
    assert feats[1, 1] == 1.0
