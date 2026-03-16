"""Tests for the 1-WL color refinement algorithm."""

import torch
import pytest

from backend.core.wl import wl_color_refinement, wl_test, build_example_pairs


class TestWLColorRefinement:
    def test_star_initial_colors(self):
        """Star graph: hub gets different color from leaves at step 0."""
        star = torch.zeros(6, 6)
        for i in range(1, 6):
            star[0, i] = star[i, 0] = 1.0
        history = wl_color_refinement(star, iterations=2)
        # Step 0: hub has degree 5, leaves have degree 1
        assert history[0][0] != history[0][1]
        # All leaves should have the same color
        assert len(set(history[0][1:])) == 1

    def test_cycle_all_same_color(self):
        """Cycle graph: all nodes should always have the same color."""
        adj = torch.zeros(6, 6)
        for i in range(6):
            j = (i + 1) % 6
            adj[i, j] = adj[j, i] = 1.0
        history = wl_color_refinement(adj, iterations=3)
        for step_colors in history:
            assert len(set(step_colors)) == 1

    def test_path_endpoint_vs_interior(self):
        """Path graph: endpoints differ from interior nodes."""
        adj = torch.zeros(6, 6)
        for i in range(5):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
        history = wl_color_refinement(adj, iterations=1)
        # Step 0: endpoints have degree 1, interior have degree 2
        assert history[0][0] != history[0][1]
        assert history[0][0] == history[0][5]  # both endpoints
        assert history[0][1] == history[0][2]  # interior nodes

    def test_color_count_bounded(self):
        """Number of distinct colors should not exceed node count."""
        adj = torch.zeros(8, 8)
        for i in range(7):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
        history = wl_color_refinement(adj, iterations=5)
        for step_colors in history:
            assert len(set(step_colors)) <= 8

    def test_refinement_monotonic(self):
        """Refinement should never merge previously distinct colors."""
        star = torch.zeros(6, 6)
        for i in range(1, 6):
            star[0, i] = star[i, 0] = 1.0
        history = wl_color_refinement(star, iterations=3)
        for t in range(len(history) - 1):
            # If two nodes had different colors at step t,
            # they must have different colors at step t+1
            for i in range(6):
                for j in range(i + 1, 6):
                    if history[t][i] != history[t][j]:
                        assert history[t + 1][i] != history[t + 1][j]


class TestWLTest:
    def test_star_vs_path_distinguished(self):
        """WL should distinguish Star K1,5 from Path P6."""
        pairs = build_example_pairs()
        result = wl_test(pairs[0]["graphA"]["adj"], pairs[0]["graphB"]["adj"])
        assert result["distinguished"] is True
        assert result["distinguishing_iteration"] == 0

    def test_c6_vs_c3c3_not_distinguished(self):
        """WL should NOT distinguish C6 from C3+C3 (both 2-regular)."""
        pairs = build_example_pairs()
        result = wl_test(pairs[1]["graphA"]["adj"], pairs[1]["graphB"]["adj"])
        assert result["distinguished"] is False

    def test_c8_vs_c4c4_not_distinguished(self):
        """WL should NOT distinguish C8 from C4+C4 (both 2-regular)."""
        pairs = build_example_pairs()
        result = wl_test(pairs[2]["graphA"]["adj"], pairs[2]["graphB"]["adj"])
        assert result["distinguished"] is False

    def test_identical_graphs(self):
        """Identical graphs should have matching histograms."""
        adj = torch.zeros(5, 5)
        for i in range(5):
            j = (i + 1) % 5
            adj[i, j] = adj[j, i] = 1.0
        result = wl_test(adj, adj.clone())
        assert result["distinguished"] is False

    def test_iterations_count(self):
        """Result should have correct number of iteration steps."""
        pairs = build_example_pairs()
        result = wl_test(pairs[0]["graphA"]["adj"], pairs[0]["graphB"]["adj"],
                         iterations=5)
        assert len(result["iterations"]) == 6  # 0..5


class TestBuildExamplePairs:
    def test_returns_three_pairs(self):
        pairs = build_example_pairs()
        assert len(pairs) == 3

    def test_adjacency_symmetric(self):
        """All adjacency matrices should be symmetric."""
        pairs = build_example_pairs()
        for pair in pairs:
            for key in ["graphA", "graphB"]:
                adj = pair[key]["adj"]
                assert torch.allclose(adj, adj.t())

    def test_no_self_loops(self):
        """No self-loops in any graph."""
        pairs = build_example_pairs()
        for pair in pairs:
            for key in ["graphA", "graphB"]:
                adj = pair[key]["adj"]
                assert (adj.diagonal() == 0).all()
