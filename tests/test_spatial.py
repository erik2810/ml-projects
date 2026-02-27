"""Tests for spatial graph generation modules.

Covers: SpatialGraph data structure, synthetic generators,
metrics, spatial tree VAE, and joint diffusion.
"""

import torch
import pytest
import math

from backend.core.spatial.graph3d import SpatialGraph, parse_swc, to_swc
from backend.core.spatial.synthetic import random_branching_tree, random_neuron_morphology
from backend.core.spatial.metrics import (
    sholl_analysis,
    strahler_numbers,
    tree_edit_distance,
    morphological_features,
    spatial_graph_mmd,
    full_evaluation,
)
from backend.core.spatial.tree_gen import SpatialTreeVAE, train_spatial_vae
from backend.core.spatial.diffusion3d import (
    SpatialGraphDiffusion,
    train_spatial_diffusion,
    cosine_beta_schedule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_tree():
    """A small tree:  0 -> 1 -> 3
                       \\-> 2 -> 4
    All nodes in known positions."""
    pos = torch.tensor([
        [0.0, 0.0, 0.0],   # root
        [1.0, 0.0, 0.0],   # child of 0
        [0.0, 1.0, 0.0],   # child of 0
        [2.0, 0.0, 0.0],   # child of 1
        [0.0, 2.0, 0.0],   # child of 2
    ])
    parent = torch.tensor([-1, 0, 0, 1, 2])
    adj = torch.zeros(5, 5)
    for i in range(5):
        p = parent[i].item()
        if p >= 0:
            adj[i, int(p)] = 1.0
            adj[int(p), i] = 1.0
    return SpatialGraph(pos=pos, adj=adj, parent=parent)


@pytest.fixture
def branching_trees():
    """Small batch of random trees for distribution-level tests."""
    return [random_branching_tree(num_nodes=20, branch_prob=0.15) for _ in range(5)]


# ---------------------------------------------------------------------------
# SpatialGraph tests
# ---------------------------------------------------------------------------

class TestSpatialGraph:
    def test_num_nodes(self, simple_tree):
        assert simple_tree.num_nodes == 5

    def test_num_edges(self, simple_tree):
        assert simple_tree.num_edges == 4

    def test_root(self, simple_tree):
        assert simple_tree.root == 0

    def test_children_of_root(self, simple_tree):
        children = simple_tree.children_of(0)
        assert sorted(children) == [1, 2]

    def test_children_of_leaf(self, simple_tree):
        assert simple_tree.children_of(3) == []
        assert simple_tree.children_of(4) == []

    def test_depth_root(self, simple_tree):
        assert simple_tree.depth(0) == 0

    def test_depth_leaves(self, simple_tree):
        assert simple_tree.depth(3) == 2
        assert simple_tree.depth(4) == 2

    def test_depth_internal(self, simple_tree):
        assert simple_tree.depth(1) == 1
        assert simple_tree.depth(2) == 1

    def test_subtree_sizes(self, simple_tree):
        sizes = simple_tree.subtree_sizes()
        assert sizes[0].item() == 5   # root contains all
        assert sizes[1].item() == 2   # node 1 + node 3
        assert sizes[2].item() == 2   # node 2 + node 4
        assert sizes[3].item() == 1   # leaf
        assert sizes[4].item() == 1   # leaf

    def test_segment_lengths(self, simple_tree):
        lengths = simple_tree.segment_lengths()
        assert lengths.numel() == 4
        # edge 1->0: length 1.0, edge 2->0: length 1.0
        # edge 3->1: length 1.0, edge 4->2: length 1.0
        assert torch.allclose(lengths, torch.ones(4), atol=1e-5)

    def test_branch_angles(self, simple_tree):
        angles = simple_tree.branch_angles()
        # node 0 has two children at 90 degrees
        assert angles.numel() >= 1
        expected = math.pi / 2  # 90 degrees
        assert abs(angles[0].item() - expected) < 0.01

    def test_pad_larger(self, simple_tree):
        padded = simple_tree.pad(10)
        assert padded.pos.shape == (10, 3)
        assert padded.adj.shape == (10, 10)
        assert padded.parent.shape == (10,)
        # original data preserved
        assert torch.allclose(padded.pos[:5], simple_tree.pos)

    def test_pad_smaller(self, simple_tree):
        padded = simple_tree.pad(3)
        assert padded.pos.shape == (3, 3)

    def test_to_device(self, simple_tree):
        moved = simple_tree.to('cpu')
        assert moved.pos.device.type == 'cpu'


class TestSWC:
    def test_roundtrip(self, simple_tree):
        """to_swc -> parse_swc should preserve structure."""
        swc_text = to_swc(simple_tree)
        parsed = parse_swc(swc_text)

        assert parsed.num_nodes == simple_tree.num_nodes
        assert torch.allclose(parsed.pos, simple_tree.pos, atol=1e-3)
        # parent structure should match
        assert torch.equal(parsed.parent, simple_tree.parent)

    def test_parse_empty(self):
        parsed = parse_swc("")
        assert parsed.num_nodes == 0

    def test_parse_with_comments(self):
        swc = """# comment line
# another comment
1 1 0.0 0.0 0.0 1.0 -1
2 3 1.0 0.0 0.0 0.5 1
"""
        parsed = parse_swc(swc)
        assert parsed.num_nodes == 2
        assert parsed.parent[0].item() == -1
        assert parsed.parent[1].item() == 0


# ---------------------------------------------------------------------------
# Synthetic generator tests
# ---------------------------------------------------------------------------

class TestSynthetic:
    def test_branching_tree_returns_spatial_graph(self):
        g = random_branching_tree(num_nodes=30)
        assert isinstance(g, SpatialGraph)

    def test_branching_tree_shape(self):
        g = random_branching_tree(num_nodes=30)
        assert g.pos.shape[1] == 3
        assert g.adj.shape[0] == g.adj.shape[1] == g.num_nodes

    def test_branching_tree_is_connected(self):
        g = random_branching_tree(num_nodes=25)
        # a tree with N nodes has N-1 edges
        assert g.num_edges == g.num_nodes - 1

    def test_branching_tree_adj_symmetric(self):
        g = random_branching_tree(num_nodes=20)
        assert torch.allclose(g.adj, g.adj.t())

    def test_neuron_morphology_returns_spatial_graph(self):
        g = random_neuron_morphology(num_nodes=40)
        assert isinstance(g, SpatialGraph)

    def test_neuron_has_soma(self):
        g = random_neuron_morphology(num_nodes=40)
        assert g.node_types is not None
        assert g.node_types[0].item() == 1  # soma type

    def test_neuron_has_radii(self):
        g = random_neuron_morphology(num_nodes=40)
        assert g.radii is not None
        assert (g.radii > 0).all()

    def test_neuron_adj_symmetric(self):
        g = random_neuron_morphology(num_nodes=30)
        assert torch.allclose(g.adj, g.adj.t())


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_sholl_analysis_shape(self, simple_tree):
        radii, crossings = sholl_analysis(simple_tree, num_shells=10)
        assert radii.shape == (10,)
        assert crossings.shape == (10,)

    def test_sholl_crossings_nonnegative(self, simple_tree):
        _, crossings = sholl_analysis(simple_tree)
        assert (crossings >= 0).all()

    def test_strahler_leaves_are_one(self, simple_tree):
        orders = strahler_numbers(simple_tree)
        # leaves (nodes 3, 4) should have order 1
        assert orders[3].item() == 1
        assert orders[4].item() == 1

    def test_strahler_root_order(self, simple_tree):
        orders = strahler_numbers(simple_tree)
        # root has two children with order 1 each -> root gets 2
        # node 1 has one child with order 1 -> node 1 gets 1
        # node 2 has one child with order 1 -> node 2 gets 1
        # root has two children both with order 1 -> root gets 2
        assert orders[0].item() == 2

    def test_strahler_internal_nodes(self, simple_tree):
        orders = strahler_numbers(simple_tree)
        assert orders[1].item() == 1
        assert orders[2].item() == 1

    def test_ted_identity(self, simple_tree):
        dist = tree_edit_distance(simple_tree, simple_tree)
        # distance to self should be 0
        assert dist == 0.0 or dist < 1e-6

    def test_ted_symmetry(self):
        """Approximate TED uses greedy matching so not perfectly symmetric.
        Check that the two directions are in the same ballpark."""
        g1 = random_branching_tree(num_nodes=15)
        g2 = random_branching_tree(num_nodes=15)
        d12 = tree_edit_distance(g1, g2)
        d21 = tree_edit_distance(g2, g1)
        avg = (d12 + d21) / 2
        assert abs(d12 - d21) / (avg + 1e-8) < 0.5  # within 50% relative

    def test_ted_nonnegative(self):
        g1 = random_branching_tree(num_nodes=12)
        g2 = random_branching_tree(num_nodes=12)
        assert tree_edit_distance(g1, g2) >= 0.0

    def test_morphological_features_shape(self, simple_tree):
        feats = morphological_features(simple_tree)
        assert feats.shape == (12,)

    def test_morphological_features_values(self, simple_tree):
        feats = morphological_features(simple_tree)
        # feature[2] = num_branch_points: node 0 has 2 children -> 1 branch point
        assert feats[2].item() == 1.0
        # feature[3] = num_tips: nodes 3, 4 -> 2 tips
        assert feats[3].item() == 2.0
        # feature[4] = Strahler of root -> 2
        assert feats[4].item() == 2.0

    def test_mmd_same_distribution(self, branching_trees):
        mmd = spatial_graph_mmd(branching_trees, branching_trees)
        assert mmd < 0.1  # should be near 0 for identical sets

    def test_mmd_nonnegative(self, branching_trees):
        g2 = [random_branching_tree(num_nodes=20) for _ in range(5)]
        mmd = spatial_graph_mmd(branching_trees, g2)
        assert mmd >= 0.0

    def test_full_evaluation_keys(self, branching_trees):
        g2 = [random_branching_tree(num_nodes=20) for _ in range(5)]
        results = full_evaluation(branching_trees, g2)
        expected_keys = {
            'mmd', 'segment_length_w1', 'branch_angle_w1',
            'strahler_w1', 'feature_mse', 'sholl_profile_mae',
        }
        assert expected_keys.issubset(set(results.keys()))

    def test_full_evaluation_values_finite(self, branching_trees):
        g2 = [random_branching_tree(num_nodes=20) for _ in range(5)]
        results = full_evaluation(branching_trees, g2)
        for k, v in results.items():
            assert math.isfinite(v), f"{k} is not finite: {v}"


# ---------------------------------------------------------------------------
# Spatial Tree VAE tests
# ---------------------------------------------------------------------------

class TestSpatialTreeVAE:
    def test_forward_returns_loss(self, simple_tree):
        model = SpatialTreeVAE(latent_dim=8, hidden_dim=16, max_nodes=32)
        out = model(simple_tree)
        assert 'loss' in out
        assert 'kl' in out
        assert 'graph' in out
        assert isinstance(out['graph'], SpatialGraph)

    def test_forward_loss_is_scalar(self, simple_tree):
        model = SpatialTreeVAE(latent_dim=8, hidden_dim=16, max_nodes=32)
        out = model(simple_tree)
        assert out['loss'].dim() == 0

    def test_generate_returns_graphs(self):
        model = SpatialTreeVAE(latent_dim=8, hidden_dim=16, max_nodes=20)
        graphs = model.generate(num_samples=2)
        assert len(graphs) == 2
        for g in graphs:
            assert isinstance(g, SpatialGraph)
            assert g.num_nodes >= 1

    def test_interpolate(self, simple_tree):
        model = SpatialTreeVAE(latent_dim=8, hidden_dim=16, max_nodes=32)
        graphs = model.interpolate(simple_tree, simple_tree, steps=3)
        assert len(graphs) == 4  # steps + 1

    def test_training_runs(self):
        model = SpatialTreeVAE(latent_dim=8, hidden_dim=16, max_nodes=32)
        graphs = [random_branching_tree(num_nodes=10) for _ in range(3)]
        losses = train_spatial_vae(model, graphs, epochs=3, lr=1e-3)
        assert len(losses) == 3
        assert all(math.isfinite(l) for l in losses)


# ---------------------------------------------------------------------------
# Spatial diffusion tests
# ---------------------------------------------------------------------------

class TestSpatialDiffusion:
    def test_cosine_schedule_shape(self):
        betas = cosine_beta_schedule(50)
        assert betas.shape == (50,)

    def test_cosine_schedule_range(self):
        betas = cosine_beta_schedule(100)
        assert (betas > 0).all()
        assert (betas < 1).all()

    def test_forward_returns_losses(self):
        model = SpatialGraphDiffusion(
            max_nodes=16, hidden_dim=16, timesteps=10, num_layers=1,
        )
        g = random_branching_tree(num_nodes=10)
        out = model(g)
        assert 'loss' in out
        assert 'pos_loss' in out
        assert 'adj_loss' in out

    def test_forward_loss_is_scalar(self):
        model = SpatialGraphDiffusion(
            max_nodes=16, hidden_dim=16, timesteps=10, num_layers=1,
        )
        g = random_branching_tree(num_nodes=10)
        out = model(g)
        assert out['loss'].dim() == 0

    def test_sample_returns_graph(self):
        model = SpatialGraphDiffusion(
            max_nodes=16, hidden_dim=16, timesteps=5, num_layers=1,
        )
        g = model.sample(num_nodes=8)
        assert isinstance(g, SpatialGraph)
        assert g.pos.shape == (8, 3)
        assert g.adj.shape == (8, 8)

    def test_sample_adj_symmetric(self):
        model = SpatialGraphDiffusion(
            max_nodes=16, hidden_dim=16, timesteps=5, num_layers=1,
        )
        g = model.sample(num_nodes=8)
        assert torch.allclose(g.adj, g.adj.t())

    def test_training_runs(self):
        model = SpatialGraphDiffusion(
            max_nodes=20, hidden_dim=16, timesteps=10, num_layers=1,
        )
        graphs = [random_branching_tree(num_nodes=12) for _ in range(3)]
        losses = train_spatial_diffusion(model, graphs, epochs=2, lr=1e-3)
        assert len(losses) == 2
        assert all(math.isfinite(l) for l in losses)


# ---------------------------------------------------------------------------
# Benchmark framework tests
# ---------------------------------------------------------------------------

class TestBenchmarks:
    def test_generate_dataset(self):
        from backend.core.benchmarks.evaluation import generate_dataset, DataConfig
        config = DataConfig(regime='tree', num_train=5, num_test=2, num_nodes=15)
        train, test = generate_dataset(config)
        assert len(train) == 5
        assert len(test) == 2
        assert all(isinstance(g, SpatialGraph) for g in train)

    def test_generate_dataset_neuron(self):
        from backend.core.benchmarks.evaluation import generate_dataset, DataConfig
        config = DataConfig(regime='neuron', num_train=3, num_test=1, num_nodes=20)
        train, test = generate_dataset(config)
        assert len(train) == 3
        # neurons should have node_types
        assert train[0].node_types is not None

    def test_generate_dataset_mixed(self):
        from backend.core.benchmarks.evaluation import generate_dataset, DataConfig
        config = DataConfig(regime='mixed', num_train=4, num_test=2, num_nodes=20)
        train, test = generate_dataset(config)
        assert len(train) == 4
        assert len(test) == 2

    def test_compare_results(self):
        from backend.core.benchmarks.evaluation import compare_results, BenchmarkResult
        results = [
            BenchmarkResult(
                model_name='test_model',
                metrics={'mmd': 0.05, 'segment_length_w1': 0.1},
                train_losses=[1.0, 0.5],
                train_time=10.0,
                num_params=1000,
            )
        ]
        table = compare_results(results)
        assert 'test_model' in table
        assert 'mmd' in table
