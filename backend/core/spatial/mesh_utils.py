"""
Utilities for low-poly 3D mesh generation and manipulation.

Procedural generators produce SpatialGraph objects with cycles (general
graphs, not trees). These are used for training the mesh VAE and
demonstrating latent-space interpolation between arbitrary 3D shapes.

Two shape primitives:
    - Deformed icosahedron: 12-vertex irregular polyhedron based on
      golden-ratio coordinates with random perturbation (30 edges)
    - Low-poly rock: random points on a deformed sphere with K-NN
      adjacency — fully asymmetric, 20-40 vertices

OBJ parser is also provided for importing external low-poly models.
"""

import torch
from torch import Tensor
import math

from .graph3d import SpatialGraph


# ---------------------------------------------------------------------------
# OBJ file parsing
# ---------------------------------------------------------------------------

def parse_obj(text: str, device: torch.device | None = None) -> SpatialGraph:
    """Parse a Wavefront OBJ string into a SpatialGraph.

    Reads vertex positions (v) and face definitions (f). Faces are
    converted to edges (each face edge becomes an undirected graph edge).
    Only triangular and quad faces are supported.
    """
    vertices = []
    face_edges = set()

    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if parts[0] == 'v' and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == 'f' and len(parts) >= 4:
            # face vertex indices (OBJ is 1-indexed, may have v/vt/vn format)
            indices = []
            for p in parts[1:]:
                idx = int(p.split('/')[0]) - 1  # convert to 0-indexed
                indices.append(idx)
            # add edges for each consecutive pair + closing edge
            for i in range(len(indices)):
                a, b = indices[i], indices[(i + 1) % len(indices)]
                face_edges.add((min(a, b), max(a, b)))

    return mesh_to_spatial_graph(vertices, face_edges, device=device)


def mesh_to_spatial_graph(
    vertices: list[list[float]],
    edges: set[tuple[int, int]],
    device: torch.device | None = None,
) -> SpatialGraph:
    """Convert vertex list and edge set to a SpatialGraph.

    Parent array is set to all -1 (no tree structure for meshes).
    """
    n = len(vertices)
    pos = torch.tensor(vertices, dtype=torch.float32, device=device)

    adj = torch.zeros(n, n, dtype=torch.float32, device=device)
    for a, b in edges:
        if 0 <= a < n and 0 <= b < n:
            adj[a, b] = 1.0
            adj[b, a] = 1.0

    parent = torch.full((n,), -1, dtype=torch.long, device=device)

    return SpatialGraph(pos=pos, adj=adj, parent=parent)


# ---------------------------------------------------------------------------
# Procedural mesh generators
# ---------------------------------------------------------------------------

def deformed_icosahedron(
    scale: float = 1.0,
    noise: float = 0.15,
    device: torch.device | None = None,
) -> SpatialGraph:
    """Generate a deformed icosahedron (12 vertices, 30 edges).

    Starts from the golden-ratio icosahedron vertices, then applies
    random perturbation to break symmetry.
    """
    phi = (1 + math.sqrt(5)) / 2  # golden ratio

    # 12 vertices of a regular icosahedron
    raw = [
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ]

    pos = torch.tensor(raw, dtype=torch.float32, device=device)
    pos = pos / pos.norm(dim=1, keepdim=True) * scale  # normalize to sphere
    pos = pos + torch.randn_like(pos) * noise  # deform

    # 30 edges of the icosahedron (fixed topology)
    edges = {
        (0,1), (0,5), (0,7), (0,10), (0,11),
        (1,5), (1,7), (1,8), (1,9),
        (2,3), (2,4), (2,6), (2,10), (2,11),
        (3,4), (3,6), (3,8), (3,9),
        (4,5), (4,9), (4,11),
        (5,9), (5,11),
        (6,7), (6,8), (6,10),
        (7,8), (7,10),
        (8,9),
        (10,11),
    }

    return mesh_to_spatial_graph(pos.tolist(), edges, device=device)


def low_poly_rock(
    num_points: int = 24,
    radius: float = 1.0,
    noise: float = 0.3,
    k_neighbors: int = 6,
    device: torch.device | None = None,
) -> SpatialGraph:
    """Generate a low-poly rock shape via random sphere points + K-NN adjacency.

    Points are placed on a deformed sphere surface with random
    perturbation, then connected to their K nearest neighbors. This
    produces a fully asymmetric mesh with no regularity.
    """
    # random points on unit sphere (Marsaglia method)
    points = torch.randn(num_points, 3, device=device)
    points = points / points.norm(dim=1, keepdim=True)

    # deform: random radial scaling per point
    radial_noise = 1.0 + (torch.rand(num_points, 1, device=device) - 0.5) * noise * 2
    points = points * radial_noise * radius

    # additional random displacement
    points = points + torch.randn_like(points) * noise * 0.3

    # K-NN adjacency (pure PyTorch, no scipy)
    # compute pairwise distances
    diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N, N, 3)
    dists = diff.norm(dim=-1)  # (N, N)
    dists.fill_diagonal_(float('inf'))  # exclude self

    k = min(k_neighbors, num_points - 1)
    _, knn_idx = dists.topk(k, dim=1, largest=False)  # (N, k)

    edges = set()
    for i in range(num_points):
        for j in knn_idx[i].tolist():
            edges.add((min(i, j), max(i, j)))

    return mesh_to_spatial_graph(points.tolist(), edges, device=device)


def generate_mesh_dataset(
    num: int = 50,
    mesh_type: str = 'mixed',
    num_points_range: tuple[int, int] = (16, 32),
    device: torch.device | None = None,
) -> list[SpatialGraph]:
    """Generate a dataset of synthetic meshes for training.

    Args:
        num: number of meshes to generate
        mesh_type: 'rock', 'icosahedron', or 'mixed'
        num_points_range: (min, max) node count for rocks
        device: torch device
    """
    graphs = []
    for i in range(num):
        if mesh_type == 'icosahedron':
            g = deformed_icosahedron(
                scale=0.8 + torch.rand(1).item() * 0.4,
                noise=0.1 + torch.rand(1).item() * 0.2,
                device=device,
            )
        elif mesh_type == 'rock':
            n = torch.randint(num_points_range[0], num_points_range[1] + 1, (1,)).item()
            g = low_poly_rock(
                num_points=n,
                radius=0.8 + torch.rand(1).item() * 0.4,
                noise=0.2 + torch.rand(1).item() * 0.2,
                device=device,
            )
        else:  # mixed
            if i % 2 == 0:
                g = deformed_icosahedron(
                    scale=0.8 + torch.rand(1).item() * 0.4,
                    noise=0.1 + torch.rand(1).item() * 0.2,
                    device=device,
                )
            else:
                n = torch.randint(num_points_range[0], num_points_range[1] + 1, (1,)).item()
                g = low_poly_rock(
                    num_points=n,
                    radius=0.8 + torch.rand(1).item() * 0.4,
                    noise=0.2 + torch.rand(1).item() * 0.2,
                    device=device,
                )
        graphs.append(g)
    return graphs
