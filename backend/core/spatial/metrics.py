"""
Evaluation metrics for spatial graph (tree) generation.

The central argument: point-cloud metrics (e.g. Chamfer distance, EMD on
positions) miss the graph structure entirely. Two morphologies can have
similar point clouds but completely different connectivity, branching
patterns, and topological properties. We need metrics that are *jointly*
aware of both geometry and topology.

Implements:
    - Tree edit distance (Zhang-Shasha inspired, simplified for unordered trees)
    - Sholl analysis (radial intersection profile)
    - Branch angle and segment length distributions
    - Strahler numbers (stream ordering for branching complexity)
    - Spatial graph MMD (Maximum Mean Discrepancy using graph+geometry features)
    - Per-graph morphological feature vector (for distributional comparisons)

References:
    - Zhang & Shasha, "Simple fast algorithms for the editing distance
      between trees and related problems", SIAM J. Comput., 1989
    - Sholl, "Dendritic organization in the neurons of the visual and
      motor cortices of the cat", J. Anat., 1953
    - Cuntz et al., "One rule to grow them all", PLoS Comp Bio, 2010
    - Kanari et al., "A topological representation of branching neuronal
      morphologies", Neuroinformatics, 2018
"""

import torch
from torch import Tensor
import math

from .graph3d import SpatialGraph


# ---------------------------------------------------------------------------
# Sholl analysis
# ---------------------------------------------------------------------------

def sholl_analysis(
    graph: SpatialGraph,
    center: int | None = None,
    num_shells: int = 20,
    max_radius: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute the Sholl profile: number of edges crossing each spherical shell.

    Args:
        graph: the spatial graph
        center: index of the center node (default: root)
        num_shells: number of radial bins
        max_radius: outer radius (default: auto from data)

    Returns:
        radii: (num_shells,) center of each radial bin
        crossings: (num_shells,) number of edges intersecting each shell
    """
    if center is None:
        center = graph.root
    origin = graph.pos[center]

    # distance of each node from center
    dists = torch.norm(graph.pos - origin.unsqueeze(0), dim=1)

    if max_radius is None:
        max_radius = dists.max().item() * 1.05

    bin_width = max_radius / num_shells
    radii = torch.linspace(bin_width / 2, max_radius - bin_width / 2, num_shells,
                           device=graph.device)
    crossings = torch.zeros(num_shells, device=graph.device)

    # for each edge, check which shells it crosses
    for i in range(graph.num_nodes):
        p = int(graph.parent[i].item())
        if p < 0:
            continue
        d_child = dists[i].item()
        d_parent = dists[p].item()
        r_min, r_max = min(d_child, d_parent), max(d_child, d_parent)

        for s in range(num_shells):
            shell_r = (s + 0.5) * bin_width
            if r_min <= shell_r <= r_max:
                crossings[s] += 1

    return radii, crossings


# ---------------------------------------------------------------------------
# Strahler numbers
# ---------------------------------------------------------------------------

def strahler_numbers(graph: SpatialGraph) -> Tensor:
    """Compute Strahler stream order for each node.

    Leaf nodes get order 1. A node with children of orders o1, o2, ...:
        - if max(o_i) is unique: parent order = max(o_i)
        - if two or more children share the max: parent order = max(o_i) + 1

    This gives a measure of branching complexity. Strahler number of the root
    characterizes the overall tree complexity.
    """
    n = graph.num_nodes
    orders = torch.zeros(n, dtype=torch.long, device=graph.device)

    # process bottom-up
    processing_order = sorted(range(n), key=lambda i: graph.depth(i), reverse=True)

    for i in processing_order:
        children = graph.children_of(i)
        if not children:
            orders[i] = 1
        else:
            child_orders = [orders[c].item() for c in children]
            max_order = max(child_orders)
            count_max = child_orders.count(max_order)
            orders[i] = max_order + 1 if count_max >= 2 else max_order

    return orders


# ---------------------------------------------------------------------------
# Distribution-level comparisons
# ---------------------------------------------------------------------------

def branch_angle_distribution(graph: SpatialGraph) -> Tensor:
    """Return all branch angles in the tree (radians)."""
    return graph.branch_angles()


def segment_length_distribution(graph: SpatialGraph) -> Tensor:
    """Return all segment (edge) lengths."""
    return graph.segment_lengths()


def _wasserstein_1d(p: Tensor, q: Tensor) -> float:
    """Wasserstein-1 distance between two 1D empirical distributions.

    Uses the closed-form: W_1 = integral |F_p(x) - F_q(x)| dx,
    computed via sorted samples.
    """
    if p.numel() == 0 or q.numel() == 0:
        return 0.0
    p_sorted = torch.sort(p)[0]
    q_sorted = torch.sort(q)[0]

    # interpolate to common grid
    n = max(p_sorted.numel(), q_sorted.numel())
    p_interp = torch.nn.functional.interpolate(
        p_sorted.unsqueeze(0).unsqueeze(0).float(),
        size=n, mode='linear', align_corners=True,
    ).squeeze()
    q_interp = torch.nn.functional.interpolate(
        q_sorted.unsqueeze(0).unsqueeze(0).float(),
        size=n, mode='linear', align_corners=True,
    ).squeeze()
    return (p_interp - q_interp).abs().mean().item()


# ---------------------------------------------------------------------------
# Morphological feature vector
# ---------------------------------------------------------------------------

def morphological_features(graph: SpatialGraph) -> Tensor:
    """Extract a fixed-size feature vector summarizing morphological properties.

    Returns a 12-dimensional vector:
        [0]  num_nodes (log)
        [1]  num_edges (log)
        [2]  num_branch_points
        [3]  num_tips (leaves)
        [4]  max_branch_order (Strahler number of root)
        [5]  mean_segment_length
        [6]  std_segment_length
        [7]  mean_branch_angle
        [8]  std_branch_angle
        [9]  total_path_length (sum of all segments)
        [10] max_euclidean_extent (diameter of bounding box)
        [11] tortuosity (mean ratio of path length to euclidean distance, root to tips)
    """
    n = graph.num_nodes
    features = torch.zeros(12, device=graph.device)

    features[0] = math.log(n + 1)
    features[1] = math.log(graph.num_edges + 1)

    # branching topology
    child_counts = torch.zeros(n, dtype=torch.long, device=graph.device)
    for i in range(n):
        p = int(graph.parent[i].item())
        if p >= 0:
            child_counts[p] += 1

    features[2] = (child_counts >= 2).sum().float()
    features[3] = (child_counts == 0).sum().float()

    orders = strahler_numbers(graph)
    features[4] = orders[graph.root].float()

    seg_lens = graph.segment_lengths()
    if seg_lens.numel() > 0:
        features[5] = seg_lens.mean()
        features[6] = seg_lens.std() if seg_lens.numel() > 1 else 0.0
        features[9] = seg_lens.sum()

    angles = graph.branch_angles()
    if angles.numel() > 0:
        features[7] = angles.mean()
        features[8] = angles.std() if angles.numel() > 1 else 0.0

    # spatial extent
    if n > 1:
        bbox = graph.pos.max(dim=0).values - graph.pos.min(dim=0).values
        features[10] = bbox.norm()

    # tortuosity: ratio of path-length to Euclidean distance for each tip
    tips = (child_counts == 0).nonzero(as_tuple=True)[0]
    if tips.numel() > 0:
        tortuosities = []
        for tip in tips.tolist():
            # path length from root to tip
            path_len = 0.0
            cur = tip
            while graph.parent[cur].item() >= 0:
                p = int(graph.parent[cur].item())
                path_len += (graph.pos[cur] - graph.pos[p]).norm().item()
                cur = p
            euclidean = (graph.pos[tip] - graph.pos[graph.root]).norm().item()
            if euclidean > 1e-6:
                tortuosities.append(path_len / euclidean)
        if tortuosities:
            features[11] = sum(tortuosities) / len(tortuosities)

    return features


# ---------------------------------------------------------------------------
# Tree edit distance (simplified)
# ---------------------------------------------------------------------------

def tree_edit_distance(
    g1: SpatialGraph,
    g2: SpatialGraph,
    position_weight: float = 1.0,
    structure_weight: float = 1.0,
) -> float:
    """Approximate tree edit distance combining structural and spatial cost.

    Full Zhang-Shasha is O(n^2 m^2). We use a top-down recursive
    approximation that's O(n*m) for typical trees.

    Edit operations:
        - Insert node: cost = structure_weight
        - Delete node: cost = structure_weight
        - Relabel (move): cost = position_weight * euclidean_distance

    This jointly penalizes structural differences AND positional mismatches.
    """
    # build child lists
    def get_children(g, node):
        return g.children_of(node)

    def _ted(g1, n1, g2, n2, memo):
        key = (n1, n2)
        if key in memo:
            return memo[key]

        c1 = get_children(g1, n1)
        c2 = get_children(g2, n2)

        # cost of matching these two nodes (positional distance)
        match_cost = position_weight * (g1.pos[n1] - g2.pos[n2]).norm().item()

        if not c1 and not c2:
            # both leaves
            memo[key] = match_cost
            return match_cost

        if not c1:
            # delete all of g2's subtree
            cost = match_cost
            for c in c2:
                cost += _subtree_cost(g2, c, structure_weight)
            memo[key] = cost
            return cost

        if not c2:
            cost = match_cost
            for c in c1:
                cost += _subtree_cost(g1, c, structure_weight)
            memo[key] = cost
            return cost

        # greedy matching of children (approximate)
        # compute all pairwise costs
        cost_matrix = []
        for i, ci in enumerate(c1):
            row = []
            for j, cj in enumerate(c2):
                row.append(_ted(g1, ci, g2, cj, memo))
            cost_matrix.append(row)

        # greedy assignment
        used_j = set()
        total = match_cost
        for i in range(len(c1)):
            best_j = None
            best_cost = float('inf')
            for j in range(len(c2)):
                if j not in used_j and cost_matrix[i][j] < best_cost:
                    best_cost = cost_matrix[i][j]
                    best_j = j
            if best_j is not None:
                total += best_cost
                used_j.add(best_j)
            else:
                total += _subtree_cost(g1, c1[i], structure_weight)

        # unmatched children in g2
        for j in range(len(c2)):
            if j not in used_j:
                total += _subtree_cost(g2, c2[j], structure_weight)

        memo[key] = total
        return total

    def _subtree_cost(g, node, w):
        """Cost of inserting/deleting an entire subtree."""
        cost = w  # cost for this node
        for c in get_children(g, node):
            cost += _subtree_cost(g, c, w)
        return cost

    memo = {}
    return _ted(g1, g1.root, g2, g2.root, memo)


# ---------------------------------------------------------------------------
# Spatial graph MMD
# ---------------------------------------------------------------------------

def spatial_graph_mmd(
    graphs_p: list[SpatialGraph],
    graphs_q: list[SpatialGraph],
    kernel: str = 'rbf',
    bandwidth: float | None = None,
) -> float:
    """Maximum Mean Discrepancy between two sets of spatial graphs.

    Uses morphological feature vectors as the representation, then
    computes MMD with an RBF (Gaussian) kernel. This is proper for
    evaluating generative model quality — it captures both structural
    and geometric properties, unlike point-cloud FID.

    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    where k is the kernel and x~P, y~Q.
    """
    feats_p = torch.stack([morphological_features(g) for g in graphs_p])
    feats_q = torch.stack([morphological_features(g) for g in graphs_q])

    # normalize features to zero mean, unit variance (from combined set)
    combined = torch.cat([feats_p, feats_q], dim=0)
    mu = combined.mean(dim=0)
    std = combined.std(dim=0).clamp(min=1e-6)
    feats_p = (feats_p - mu) / std
    feats_q = (feats_q - mu) / std

    if bandwidth is None:
        # median heuristic
        all_dists = torch.cdist(combined / std.unsqueeze(0),
                                combined / std.unsqueeze(0))
        bandwidth = all_dists.median().item()
        bandwidth = max(bandwidth, 1e-3)

    def rbf_kernel(x, y):
        dists = torch.cdist(x, y)
        return torch.exp(-dists ** 2 / (2 * bandwidth ** 2))

    k_pp = rbf_kernel(feats_p, feats_p)
    k_qq = rbf_kernel(feats_q, feats_q)
    k_pq = rbf_kernel(feats_p, feats_q)

    n_p, n_q = feats_p.size(0), feats_q.size(0)

    # unbiased estimator
    mmd2 = (k_pp.sum() - k_pp.trace()) / (n_p * (n_p - 1) + 1e-8)
    mmd2 += (k_qq.sum() - k_qq.trace()) / (n_q * (n_q - 1) + 1e-8)
    mmd2 -= 2 * k_pq.mean()

    return max(0.0, mmd2.item())


# ---------------------------------------------------------------------------
# Convenience: compare two sets of graphs on all metrics
# ---------------------------------------------------------------------------

def full_evaluation(
    generated: list[SpatialGraph],
    reference: list[SpatialGraph],
) -> dict[str, float]:
    """Run the complete evaluation suite.

    Returns a dict of metric_name -> value.
    """
    results = {}

    # MMD (the headline number)
    results['mmd'] = spatial_graph_mmd(generated, reference)

    # distributional comparisons on segment lengths
    gen_lengths = torch.cat([g.segment_lengths() for g in generated])
    ref_lengths = torch.cat([g.segment_lengths() for g in reference])
    results['segment_length_w1'] = _wasserstein_1d(gen_lengths, ref_lengths)

    # branch angles
    gen_angles = torch.cat([g.branch_angles() for g in generated if g.branch_angles().numel() > 0])
    ref_angles = torch.cat([g.branch_angles() for g in reference if g.branch_angles().numel() > 0])
    results['branch_angle_w1'] = _wasserstein_1d(gen_angles, ref_angles)

    # Strahler order distribution
    gen_strahler = torch.cat([strahler_numbers(g).float() for g in generated])
    ref_strahler = torch.cat([strahler_numbers(g).float() for g in reference])
    results['strahler_w1'] = _wasserstein_1d(gen_strahler, ref_strahler)

    # average morphological features
    gen_feats = torch.stack([morphological_features(g) for g in generated])
    ref_feats = torch.stack([morphological_features(g) for g in reference])
    results['feature_mse'] = ((gen_feats.mean(0) - ref_feats.mean(0)) ** 2).mean().item()

    # mean Sholl profile difference
    sholl_diffs = []
    n_compare = min(len(generated), len(reference), 20)
    for i in range(n_compare):
        _, s_gen = sholl_analysis(generated[i])
        _, s_ref = sholl_analysis(reference[i])
        # pad to same length
        max_len = max(s_gen.numel(), s_ref.numel())
        s_gen_pad = torch.zeros(max_len)
        s_ref_pad = torch.zeros(max_len)
        s_gen_pad[:s_gen.numel()] = s_gen
        s_ref_pad[:s_ref.numel()] = s_ref
        sholl_diffs.append((s_gen_pad - s_ref_pad).abs().mean().item())
    results['sholl_profile_mae'] = sum(sholl_diffs) / max(len(sholl_diffs), 1)

    return results
