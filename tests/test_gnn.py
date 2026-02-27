import torch
import pytest

from backend.core.graph_utils import erdos_renyi, normalize_adj, adj_to_edge_index
from backend.core.gnn.layers import GCNLayer, GATLayer
from backend.core.gnn.model import NodeClassifier, GraphClassifier, generate_karate_club


def test_gcn_layer_output_shape():
    adj = erdos_renyi(10, 0.3)
    x = torch.randn(10, 8)
    layer = GCNLayer(8, 16)
    out = layer(x, adj)
    assert out.shape == (10, 16)


def test_gcn_layer_with_self_loops():
    """GCN should handle isolated nodes (self-loop normalization)."""
    adj = torch.zeros(5, 5)
    adj[0, 1] = adj[1, 0] = 1.0
    x = torch.randn(5, 4)
    layer = GCNLayer(4, 8)
    out = layer(x, adj)
    assert out.shape == (5, 8)
    # isolated nodes should still produce output (via self-loop)
    assert not torch.isnan(out).any()


def test_gat_layer_single_head():
    adj = erdos_renyi(8, 0.4)
    x = torch.randn(8, 6)
    layer = GATLayer(6, 4, n_heads=1, concat=False)
    out = layer(x, adj)
    assert out.shape == (8, 4)


def test_gat_layer_multi_head_concat():
    adj = erdos_renyi(8, 0.4)
    x = torch.randn(8, 6)
    layer = GATLayer(6, 4, n_heads=3, concat=True)
    out = layer(x, adj)
    assert out.shape == (8, 12)  # 3 heads * 4 dim


def test_gat_layer_multi_head_average():
    adj = erdos_renyi(8, 0.4)
    x = torch.randn(8, 6)
    layer = GATLayer(6, 4, n_heads=3, concat=False)
    out = layer(x, adj)
    assert out.shape == (8, 4)  # averaged


def test_node_classifier_gcn():
    adj, features, labels = generate_karate_club()
    model = NodeClassifier(in_features=features.size(1), hidden=16, num_classes=2, n_layers=2)
    logits = model(features, adj)
    assert logits.shape == (34, 2)


def test_node_classifier_gat():
    adj, features, labels = generate_karate_club()
    model = NodeClassifier(
        in_features=features.size(1), hidden=8, num_classes=2,
        n_layers=2, layer_type="gat", n_heads=2,
    )
    logits = model(features, adj)
    assert logits.shape == (34, 2)


def test_graph_classifier():
    adjs = [erdos_renyi(n, 0.3) for n in [8, 10, 12]]
    features = [torch.randn(a.size(0), 4) for a in adjs]
    model = GraphClassifier(in_features=4, hidden=8, num_classes=3, n_layers=2)
    logits = model(features, adjs)
    assert logits.shape == (3, 3)


def test_karate_club_structure():
    adj, features, labels = generate_karate_club()
    assert adj.shape == (34, 34)
    assert labels.shape == (34,)
    # should be symmetric
    assert torch.allclose(adj, adj.t())
    # two communities
    assert set(labels.tolist()) == {0, 1}
