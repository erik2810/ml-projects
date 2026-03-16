"""API routes for Weisfeiler-Leman (1-WL) color refinement demonstration."""

from fastapi import APIRouter
from pydantic import BaseModel

from backend.core.wl import wl_color_refinement, wl_test, build_example_pairs
from backend.core.graph_utils import spectral_layout_2d, adj_to_edge_index

router = APIRouter(prefix="/wl", tags=["wl"])


class WLTestRequest(BaseModel):
    pair_index: int = 0
    iterations: int = 3


def _circular_layout(n):
    """Simple circular layout for regular graphs."""
    import math
    return [[math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)] for i in range(n)]


def _star_layout(n):
    """Hub at center, leaves in a circle."""
    import math
    positions = [[0.0, 0.0]]  # hub
    for i in range(1, n):
        angle = 2 * math.pi * (i - 1) / (n - 1)
        positions.append([math.cos(angle), math.sin(angle)])
    return positions


def _linear_layout(n):
    """Nodes in a horizontal line."""
    return [[2.0 * i / max(n - 1, 1) - 1.0, 0.0] for i in range(n)]


def _disconnected_layout(adj, component_sizes):
    """Layout for disconnected graphs: layout each component, offset horizontally."""
    import math
    positions = []
    offset = 0.0
    start = 0
    for size in component_sizes:
        for i in range(size):
            angle = 2 * math.pi * i / size
            positions.append([offset + math.cos(angle), math.sin(angle)])
        offset += 3.0
        start += size
    # Normalize to [-1, 1]
    if positions:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        rx = max_x - min_x or 1
        ry = max_y - min_y or 1
        for p in positions:
            p[0] = 2.0 * (p[0] - min_x) / rx - 1.0
            p[1] = 2.0 * (p[1] - min_y) / ry - 1.0
    return positions


def _graph_to_response(adj, name, layout_type="spectral", component_sizes=None):
    """Convert adjacency matrix to API response with positions and edges."""
    n = adj.size(0)
    ei = adj_to_edge_index(adj)
    edges = []
    for k in range(ei.size(1)):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i < j:
            edges.append([i, j])

    if layout_type == "star":
        positions = _star_layout(n)
    elif layout_type == "linear":
        positions = _linear_layout(n)
    elif layout_type == "circular":
        positions = _circular_layout(n)
    elif layout_type == "disconnected" and component_sizes:
        positions = _disconnected_layout(adj, component_sizes)
    else:
        positions = spectral_layout_2d(adj).tolist()

    return {
        "name": name,
        "num_nodes": n,
        "edges": edges,
        "positions": positions,
    }


@router.get("/examples")
def get_examples():
    """Return all 3 example pairs with positions, edges, and WL test results."""
    pairs = build_example_pairs()

    # Layout types per pair
    layout_configs = [
        # Pair 0: Star K1,5 vs Path P6
        {"a_layout": "star", "b_layout": "linear", "a_comp": None, "b_comp": None},
        # Pair 1: C6+short chord vs C6+long chord
        {"a_layout": "circular", "b_layout": "circular", "a_comp": None, "b_comp": None},
        # Pair 2: C6 vs C3+C3
        {"a_layout": "circular", "b_layout": "disconnected", "a_comp": None, "b_comp": [3, 3]},
    ]

    iters = 4
    results = []
    for i, pair in enumerate(pairs):
        cfg = layout_configs[i]
        adj_a = pair["graphA"]["adj"]
        adj_b = pair["graphB"]["adj"]

        # Per-graph independent WL for visualization
        history_a = wl_color_refinement(adj_a, iterations=iters)
        history_b = wl_color_refinement(adj_b, iterations=iters)
        # Disjoint-union WL for comparison
        wl_result = wl_test(adj_a, adj_b, iterations=iters)

        graph_a = _graph_to_response(
            adj_a, pair["graphA"]["name"],
            layout_type=cfg["a_layout"], component_sizes=cfg["a_comp"],
        )
        graph_b = _graph_to_response(
            adj_b, pair["graphB"]["name"],
            layout_type=cfg["b_layout"], component_sizes=cfg["b_comp"],
        )

        iterations = []
        for step_idx in range(len(history_a)):
            union_step = wl_result["iterations"][step_idx]
            iterations.append({
                "step": step_idx,
                "colors_a": history_a[step_idx],
                "colors_b": history_b[step_idx],
                "histogram_a": union_step["histogram_a"],
                "histogram_b": union_step["histogram_b"],
                "histograms_match": union_step["histograms_match"],
                "num_colors_a": len(set(history_a[step_idx])),
                "num_colors_b": len(set(history_b[step_idx])),
            })

        results.append({
            "name": pair["name"],
            "description": pair.get("description", ""),
            "graphA": graph_a,
            "graphB": graph_b,
            "distinguished": wl_result["distinguished"],
            "distinguishing_iteration": wl_result["distinguishing_iteration"],
            "iterations": iterations,
        })

    return {"pairs": results}


@router.post("/test")
def run_wl_test(req: WLTestRequest):
    """Run WL test on a specific pair with custom iteration count."""
    pairs = build_example_pairs()
    if req.pair_index < 0 or req.pair_index >= len(pairs):
        from fastapi import HTTPException
        raise HTTPException(400, f"pair_index must be 0..{len(pairs) - 1}")

    pair = pairs[req.pair_index]
    result = wl_test(pair["graphA"]["adj"], pair["graphB"]["adj"], iterations=req.iterations)

    iterations = []
    for step in result["iterations"]:
        iterations.append({
            "step": step["step"],
            "colors_a": step["colors_a"],
            "colors_b": step["colors_b"],
            "histogram_a": step["histogram_a"],
            "histogram_b": step["histogram_b"],
            "histograms_match": step["histograms_match"],
        })

    return {
        "name": pair["name"],
        "distinguished": result["distinguished"],
        "distinguishing_iteration": result["distinguishing_iteration"],
        "iterations": iterations,
    }
