"""
Synthetic generation of 3D tree morphologies.

Produces training data for spatial graph generative models. Two regimes:
  1. random_branching_tree — generic stochastic branching in 3D
  2. random_neuron_morphology — biologically-inspired neuron-like shapes
     with apical/basal dendrite-like structure

The branching process is inspired by Cuntz et al., "One Rule to Grow Them
All" (PLoS Comp Bio 2010), simplified: we use a stochastic growth cone model
with directional persistence and random branching.
"""

import torch
from torch import Tensor
import math

from .graph3d import SpatialGraph


def random_branching_tree(
    num_nodes: int = 50,
    branch_prob: float = 0.15,
    segment_length: float = 1.0,
    length_std: float = 0.3,
    direction_noise: float = 0.4,
    gravitropism: float = 0.0,
    device: torch.device | None = None,
) -> SpatialGraph:
    """Generate a 3D tree via stochastic growth.

    Each active tip extends by one segment per step. At each step,
    a tip can branch (creating a sibling growth direction) with
    probability `branch_prob`.

    Args:
        num_nodes: target node count (approximate, stops when reached)
        branch_prob: probability of bifurcation at each tip per step
        segment_length: mean segment length
        length_std: standard deviation of segment length
        direction_noise: controls angular wander (higher = more tortuous)
        gravitropism: bias toward negative y-axis (for botanical trees)
        device: torch device
    """
    pos = torch.zeros(num_nodes, 3, device=device)
    parent = torch.full((num_nodes,), -1, dtype=torch.long, device=device)

    # initial growth direction: roughly upward
    initial_dir = torch.tensor([0.0, 1.0, 0.0], device=device)

    # active tips: list of (node_index, direction_vector)
    tips = [(0, initial_dir.clone())]
    n = 1  # number of placed nodes

    while n < num_nodes and tips:
        new_tips = []
        for tip_idx, direction in tips:
            if n >= num_nodes:
                break

            # sample segment length
            seg_len = max(0.1, segment_length + length_std * torch.randn(1, device=device).item())

            # perturb direction
            noise = torch.randn(3, device=device) * direction_noise
            new_dir = direction + noise
            if gravitropism != 0.0:
                new_dir[1] -= gravitropism  # bias downward for roots, upward for shoots
            new_dir = new_dir / (new_dir.norm() + 1e-8)

            # place child
            child_pos = pos[tip_idx] + new_dir * seg_len
            pos[n] = child_pos
            parent[n] = tip_idx

            new_tips.append((n, new_dir))
            n += 1

            # branching
            if n < num_nodes and torch.rand(1, device=device).item() < branch_prob:
                branch_dir = _random_branch_direction(new_dir, device=device)
                branch_len = max(0.1, segment_length + length_std * torch.randn(1, device=device).item())
                branch_pos = pos[tip_idx] + branch_dir * branch_len
                pos[n] = branch_pos
                parent[n] = tip_idx
                new_tips.append((n, branch_dir))
                n += 1

        tips = new_tips

    # trim to actual size
    pos = pos[:n]
    parent = parent[:n]

    # build adjacency
    adj = torch.zeros(n, n, device=device)
    for i in range(n):
        p = int(parent[i].item())
        if p >= 0:
            adj[i, p] = 1.0
            adj[p, i] = 1.0

    return SpatialGraph(pos=pos, adj=adj, parent=parent)


def random_neuron_morphology(
    num_nodes: int = 80,
    num_dendrites: int = 4,
    apical_prob: float = 0.25,
    soma_radius: float = 5.0,
    segment_length: float = 8.0,
    branch_prob: float = 0.2,
    taper: float = 0.95,
    device: torch.device | None = None,
) -> SpatialGraph:
    """Generate a neuron-like morphology with soma, dendrites, and optional apical trunk.

    Loosely inspired by L5 pyramidal neuron structure:
    - Central soma
    - Multiple basal dendrites radiating outward
    - Optional apical dendrite extending upward with oblique branches

    Args:
        num_nodes: approximate total node count
        num_dendrites: number of primary dendrite branches from soma
        apical_prob: probability that one dendrite is apical (tall, upward)
        soma_radius: distance of first branch points from soma center
        segment_length: mean segment length for dendrites
        branch_prob: probability of branching per growth step
        taper: how much segment length decreases with each generation
        device: torch device
    """
    all_pos = [torch.zeros(3, device=device)]  # soma at origin
    all_parent = [-1]
    all_types = [1]  # SWC type 1 = soma
    n = 1

    nodes_per_dendrite = max(3, (num_nodes - 1) // num_dendrites)
    has_apical = torch.rand(1).item() < apical_prob

    for d in range(num_dendrites):
        is_apical = has_apical and d == 0

        # primary direction
        if is_apical:
            direction = torch.tensor([0.0, 1.0, 0.0], device=device)
            ntype = 4  # apical
            d_branch_prob = branch_prob * 0.6  # apical branches less
            d_seg_len = segment_length * 1.5
        else:
            # radially distributed in xz-plane, slight downward
            angle = 2 * math.pi * d / num_dendrites
            direction = torch.tensor([
                math.cos(angle), -0.3, math.sin(angle)
            ], device=device)
            direction = direction / direction.norm()
            ntype = 3  # basal
            d_branch_prob = branch_prob
            d_seg_len = segment_length

        # place first segment from soma
        first_pos = all_pos[0] + direction * soma_radius
        all_pos.append(first_pos)
        all_parent.append(0)
        all_types.append(ntype)
        root_of_dendrite = n
        n += 1

        # grow this dendrite
        tips = [(root_of_dendrite, direction, d_seg_len)]
        placed = 1

        while placed < nodes_per_dendrite and tips and n < num_nodes:
            new_tips = []
            for tip, d_dir, seg in tips:
                if placed >= nodes_per_dendrite or n >= num_nodes:
                    break

                noise = torch.randn(3, device=device) * 0.35
                new_dir = d_dir + noise
                new_dir = new_dir / (new_dir.norm() + 1e-8)

                child_pos = all_pos[tip] + new_dir * seg
                all_pos.append(child_pos)
                all_parent.append(tip)
                all_types.append(ntype)
                new_tips.append((n, new_dir, seg * taper))
                n += 1
                placed += 1

                # branch
                if placed < nodes_per_dendrite and n < num_nodes:
                    if torch.rand(1, device=device).item() < d_branch_prob:
                        br_dir = _random_branch_direction(new_dir, device=device)
                        br_pos = all_pos[tip] + br_dir * seg * 0.8
                        all_pos.append(br_pos)
                        all_parent.append(tip)
                        all_types.append(ntype)
                        new_tips.append((n, br_dir, seg * taper))
                        n += 1
                        placed += 1

            tips = new_tips

    # assemble tensors
    pos = torch.stack(all_pos[:n])
    parent = torch.tensor(all_parent[:n], dtype=torch.long, device=device)
    node_types = torch.tensor(all_types[:n], dtype=torch.long, device=device)

    adj = torch.zeros(n, n, device=device)
    for i in range(n):
        p = int(parent[i].item())
        if p >= 0:
            adj[i, p] = 1.0
            adj[p, i] = 1.0

    # approximate radii: thicker near soma, thinner at tips
    depths = torch.zeros(n, device=device)
    for i in range(n):
        d = 0
        cur = i
        while all_parent[cur] != -1:
            cur = all_parent[cur]
            d += 1
        depths[i] = d
    max_depth = depths.max().clamp(min=1)
    radii = 2.0 * (1.0 - 0.8 * depths / max_depth)
    radii[0] = soma_radius  # soma is larger

    return SpatialGraph(
        pos=pos, adj=adj, parent=parent,
        radii=radii, node_types=node_types,
    )


def _random_branch_direction(parent_dir: Tensor, min_angle: float = 0.4,
                             max_angle: float = 1.2,
                             device: torch.device | None = None) -> Tensor:
    """Sample a branch direction that deviates from the parent by a random angle.

    The angle is uniformly sampled from [min_angle, max_angle] radians.
    The rotation axis is random and perpendicular to parent_dir.
    """
    angle = min_angle + (max_angle - min_angle) * torch.rand(1, device=device).item()

    # random perpendicular axis via cross product with arbitrary vector
    arbitrary = torch.randn(3, device=device)
    axis = torch.linalg.cross(parent_dir, arbitrary)
    if axis.norm() < 1e-6:
        arbitrary = torch.tensor([1.0, 0.0, 0.0], device=device)
        axis = torch.linalg.cross(parent_dir, arbitrary)
    axis = axis / (axis.norm() + 1e-8)

    # Rodrigues' rotation formula
    branch = (parent_dir * math.cos(angle)
              + torch.linalg.cross(axis, parent_dir) * math.sin(angle)
              + axis * torch.dot(axis, parent_dir) * (1 - math.cos(angle)))
    return branch / (branch.norm() + 1e-8)
