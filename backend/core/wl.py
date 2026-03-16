"""
1-dimensional Weisfeiler-Leman (1-WL) color refinement algorithm.

The 1-WL test is a classical graph isomorphism heuristic that iteratively
refines node colors by aggregating multisets of neighbor colors. It is
provably equivalent in discriminative power to message-passing GNNs
(Xu et al., 2019; Morris et al., 2019).

This module provides:
    - Color refinement on a single graph
    - Pairwise WL isomorphism test with per-iteration diagnostics
    - Pre-built example pairs demonstrating success and failure cases
"""

import torch
from torch import Tensor
from collections import Counter


def wl_color_refinement(adj: Tensor, iterations: int = 3) -> list[list[int]]:
    """Run 1-WL color refinement on a graph.

    Initializes node colors from degree, then iteratively refines by
    hashing (color, sorted_neighbor_colors) and relabeling to consecutive
    integers.

    Args:
        adj: (N, N) symmetric adjacency matrix.
        iterations: Number of refinement iterations.

    Returns:
        List of length ``iterations + 1``, where each entry is a list of
        per-node color integers (iteration 0 is the initial degree coloring).
    """
    n = adj.size(0)
    colors = adj.sum(dim=1).long().tolist()
    history: list[list[int]] = [_relabel(colors)]

    for _ in range(iterations):
        colors = history[-1]
        new_raw: list[tuple] = []
        for i in range(n):
            neighbors = torch.where(adj[i] > 0)[0].tolist()
            neighbor_colors = tuple(sorted(colors[j] for j in neighbors))
            new_raw.append((colors[i], neighbor_colors))
        history.append(_relabel_tuples(new_raw))

    return history


def wl_test(
    adj1: Tensor, adj2: Tensor, iterations: int = 3,
) -> dict:
    """Compare two graphs using the 1-WL test.

    Runs color refinement on the disjoint union of both graphs so that
    colors are in a shared namespace, then compares color histograms
    at each iteration.

    Returns:
        dict with keys:
            - iterations: list of per-step dicts with colors and histograms
            - distinguished: bool (True if histograms differ at any step)
            - distinguishing_iteration: int or None
    """
    n1 = adj1.size(0)
    n2 = adj2.size(0)

    # Build disjoint union adjacency (block diagonal)
    n = n1 + n2
    adj_union = torch.zeros(n, n, device=adj1.device)
    adj_union[:n1, :n1] = adj1
    adj_union[n1:, n1:] = adj2

    # Run WL on the union — colors are in a shared namespace
    union_history = wl_color_refinement(adj_union, iterations)

    steps = []
    distinguished = False
    distinguishing_iter = None

    for step, colors in enumerate(union_history):
        c1 = colors[:n1]
        c2 = colors[n1:]
        hist1 = dict(Counter(c1))
        hist2 = dict(Counter(c2))
        same = hist1 == hist2
        if not same and not distinguished:
            distinguished = True
            distinguishing_iter = step
        steps.append({
            "step": step,
            "colors_a": c1,
            "colors_b": c2,
            "histogram_a": hist1,
            "histogram_b": hist2,
            "histograms_match": same,
        })

    return {
        "iterations": steps,
        "distinguished": distinguished,
        "distinguishing_iteration": distinguishing_iter,
    }


def build_example_pairs() -> list[dict]:
    """Construct three example graph pairs for WL demonstration.

    Pair 0: Star K(1,5) vs Path P6 — WL succeeds at iteration 0
            (different degree distributions → immediate distinction)
    Pair 1: C6+short chord vs C6+long chord — WL succeeds at iteration 1
            (same degree sequence, but neighborhoods differ after refinement)
    Pair 2: C6 vs C3+C3 — WL fails (both 2-regular, fundamental limit)
    """
    pairs = []

    # Pair 0: Star K_{1,5} vs Path P_6
    star = torch.zeros(6, 6)
    for i in range(1, 6):
        star[0, i] = star[i, 0] = 1.0
    path = torch.zeros(6, 6)
    for i in range(5):
        path[i, i + 1] = path[i + 1, i] = 1.0
    pairs.append({
        "name": "Star K\u2081,\u2085 vs Path P\u2086",
        "graphA": {"adj": star, "name": "Star K\u2081,\u2085"},
        "graphB": {"adj": path, "name": "Path P\u2086"},
        "description": "Different degree distributions make these trivially distinguishable.",
    })

    # Pair 1: C6 + short chord (0-2) vs C6 + long chord (0-3)
    # Both have degree sequence [2,2,2,2,3,3] — identical at iteration 0.
    # At iteration 1, the short chord creates triangles while the long chord
    # does not, causing divergent color refinement.
    c6_short = torch.zeros(6, 6)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2)]:
        c6_short[i, j] = c6_short[j, i] = 1.0
    c6_long = torch.zeros(6, 6)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]:
        c6_long[i, j] = c6_long[j, i] = 1.0
    pairs.append({
        "name": "C\u2086+short chord vs C\u2086+long chord",
        "graphA": {"adj": c6_short, "name": "C\u2086+e(0,2)"},
        "graphB": {"adj": c6_long, "name": "C\u2086+e(0,3)"},
        "description": "Same degree sequence [2,2,2,2,3,3]. Neighborhood refinement reveals structural difference.",
    })

    # Pair 2: C6 (hexagon) vs C3 + C3 (two triangles)
    c6 = _make_cycle(6)
    c3c3 = torch.zeros(6, 6)
    for i, j in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]:
        c3c3[i, j] = c3c3[j, i] = 1.0
    pairs.append({
        "name": "C\u2086 vs C\u2083\u222aC\u2083",
        "graphA": {"adj": c6, "name": "C\u2086 (hexagon)"},
        "graphB": {"adj": c3c3, "name": "C\u2083\u222AC\u2083 (two triangles)"},
        "description": "Both 2-regular \u2014 all nodes identical to WL. This is the fundamental expressivity limit of 1-WL and message-passing GNNs.",
    })

    return pairs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_cycle(n: int) -> Tensor:
    """Create adjacency matrix for cycle graph C_n."""
    adj = torch.zeros(n, n)
    for i in range(n):
        j = (i + 1) % n
        adj[i, j] = adj[j, i] = 1.0
    return adj


def _relabel(colors: list[int]) -> list[int]:
    """Relabel colors to consecutive integers starting from 0."""
    mapping: dict[int, int] = {}
    counter = 0
    result = []
    for c in colors:
        if c not in mapping:
            mapping[c] = counter
            counter += 1
        result.append(mapping[c])
    return result


def _relabel_tuples(raw: list[tuple]) -> list[int]:
    """Relabel tuple-based colors to consecutive integers."""
    mapping: dict[tuple, int] = {}
    counter = 0
    result = []
    for t in raw:
        if t not in mapping:
            mapping[t] = counter
            counter += 1
        result.append(mapping[t])
    return result
