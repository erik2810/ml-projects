"""
Utilities for low-poly 3D mesh generation and manipulation.

Procedural generators produce SpatialGraph objects with cycles (general
graphs, not trees). These are used for training the mesh VAE and
demonstrating latent-space interpolation between arbitrary 3D shapes.

Six showcase primitives:
    - Cube (8 vertices, 12 edges)
    - Octahedron (6 vertices, 12 edges)
    - Deformed icosahedron (12 vertices, 30 edges)
    - Hexagonal prism (12 vertices, 18 edges)
    - 3D star / stellated dodecahedron (14 vertices, 24 edges)
    - Low-poly torus (32 vertices, 64 edges)

Additional generators for training data:
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

def cube(
    scale: float = 1.0,
    noise: float = 0.08,
    device: torch.device | None = None,
) -> SpatialGraph:
    """Generate a deformed cube (8 vertices, 12 edges)."""
    raw = [
        [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
        [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
    ]
    pos = torch.tensor(raw, dtype=torch.float32, device=device) * scale
    pos = pos + torch.randn_like(pos) * noise
    edges = {
        (0,1), (0,2), (0,4), (1,3), (1,5), (2,3),
        (2,6), (3,7), (4,5), (4,6), (5,7), (6,7),
    }
    return mesh_to_spatial_graph(pos.tolist(), edges, device=device)


def octahedron(
    scale: float = 1.0,
    noise: float = 0.06,
    device: torch.device | None = None,
) -> SpatialGraph:
    """Generate a deformed octahedron (6 vertices, 12 edges)."""
    raw = [
        [ 1, 0, 0], [-1, 0, 0], [0,  1, 0],
        [0, -1, 0], [ 0, 0, 1], [0, 0, -1],
    ]
    pos = torch.tensor(raw, dtype=torch.float32, device=device) * scale
    pos = pos + torch.randn_like(pos) * noise
    edges = {
        (0,2), (0,3), (0,4), (0,5),
        (1,2), (1,3), (1,4), (1,5),
        (2,4), (2,5), (3,4), (3,5),
    }
    return mesh_to_spatial_graph(pos.tolist(), edges, device=device)


def hexagonal_prism(
    scale: float = 1.0,
    noise: float = 0.06,
    device: torch.device | None = None,
) -> SpatialGraph:
    """Generate a deformed hexagonal prism (12 vertices, 18 edges).

    Bottom hexagon (nodes 0-5) and top hexagon (nodes 6-11) connected
    by vertical struts.
    """
    verts = []
    for ring_y, offset in [(-1.0, 0), (1.0, 6)]:
        for k in range(6):
            angle = math.pi / 3 * k
            x = math.cos(angle) * scale
            z = math.sin(angle) * scale
            verts.append([x, ring_y * scale, z])

    pos = torch.tensor(verts, dtype=torch.float32, device=device)
    pos = pos + torch.randn_like(pos) * noise

    edges = set()
    for i in range(6):
        edges.add((i, (i + 1) % 6))          # bottom ring
        edges.add((6 + i, 6 + (i + 1) % 6))  # top ring
        edges.add((i, 6 + i))                 # vertical struts
    return mesh_to_spatial_graph(pos.tolist(), edges, device=device)


def star_3d(
    scale: float = 1.0,
    noise: float = 0.04,
    device: torch.device | None = None,
) -> SpatialGraph:
    """Generate a 3D star shape (14 vertices, 24 edges).

    An outer ring of 6 tip vertices connected through a top and bottom
    apex, forming a stellated shape.
    """
    verts = []
    # 6 outer tip vertices on a ring (alternating radii for star shape)
    for k in range(6):
        angle = math.pi / 3 * k
        r = scale * (1.1 if k % 2 == 0 else 0.4)
        x = math.cos(angle) * r
        z = math.sin(angle) * r
        y = (0.38 if k % 2 == 0 else -0.38) * scale
        verts.append([x, y, z])
    # 6 inner ring vertices (half radius)
    for k in range(6):
        angle = math.pi / 3 * k
        r = scale * (0.4 if k % 2 == 0 else 0.35)
        x = math.cos(angle) * r
        z = math.sin(angle) * r
        y = (-0.38 if k % 2 == 0 else 0.38) * scale
        verts.append([x, y, z])
    # top and bottom apex
    verts.append([0,  0.55 * scale, 0])  # node 12 = top
    verts.append([0, -0.55 * scale, 0])  # node 13 = bottom

    pos = torch.tensor(verts, dtype=torch.float32, device=device)
    pos = pos + torch.randn_like(pos) * noise

    edges = set()
    # outer ring
    for k in range(6):
        edges.add((k, (k + 1) % 6))
        edges.add((6 + k, 6 + (k + 1) % 6))
    # connect even outer tips to top apex, odd to bottom
    for k in range(0, 6, 2):
        edges.add((k, 12))
        edges.add((k, 13))
    for k in range(1, 6, 2):
        edges.add((k, 12))
        edges.add((k, 13))
    return mesh_to_spatial_graph(pos.tolist(), edges, device=device)


def low_poly_torus(
    major_r: float = 1.0,
    minor_r: float = 0.35,
    n_major: int = 8,
    n_minor: int = 4,
    noise: float = 0.04,
    device: torch.device | None = None,
) -> SpatialGraph:
    """Generate a low-poly torus (n_major * n_minor vertices).

    Default: 8×4 = 32 vertices, 64 edges. Each ring of n_minor vertices
    is connected to the next ring, wrapping around for both loops.
    """
    verts = []
    for i in range(n_major):
        theta = 2 * math.pi * i / n_major
        cx = major_r * math.cos(theta)
        cz = major_r * math.sin(theta)
        for j in range(n_minor):
            phi = 2 * math.pi * j / n_minor
            x = (major_r + minor_r * math.cos(phi)) * math.cos(theta)
            y = minor_r * math.sin(phi)
            z = (major_r + minor_r * math.cos(phi)) * math.sin(theta)
            verts.append([x, y, z])

    pos = torch.tensor(verts, dtype=torch.float32, device=device)
    pos = pos + torch.randn_like(pos) * noise

    edges = set()
    total = n_major * n_minor
    for i in range(n_major):
        for j in range(n_minor):
            idx = i * n_minor + j
            # connect within ring
            next_j = i * n_minor + (j + 1) % n_minor
            edges.add((min(idx, next_j), max(idx, next_j)))
            # connect to next ring
            next_i = ((i + 1) % n_major) * n_minor + j
            edges.add((min(idx, next_i), max(idx, next_i)))
    return mesh_to_spatial_graph(pos.tolist(), edges, device=device)


def geometric_interpolate(
    g1: SpatialGraph,
    g2: SpatialGraph,
    steps: int = 7,
) -> list[SpatialGraph]:
    """Smoothly interpolate between two meshes via direct geometry blending.

    Instead of decoding from a learned latent space (which requires a
    well-trained VAE), this function produces deterministic, visually
    smooth morphs by:

        1. Padding both meshes to the same node count.
        2. Greedy nearest-neighbour matching for node correspondence
           (centred, so shape orientation is respected).
        3. Linear position interpolation between matched nodes.
        4. Gradual edge topology transition: shared edges persist,
           source-only edges are progressively removed, and
           target-only edges are progressively added.

    Returns ``steps + 1`` frames (including endpoints at t=0 and t=1).
    """
    device = g1.pos.device
    n1, n2 = g1.num_nodes, g2.num_nodes
    N = max(n1, n2)

    # --- pad positions (phantom nodes at centroid) ---
    def _pad_pos(pos, n):
        if n >= N:
            return pos.clone().detach()
        centroid = pos.mean(0, keepdim=True)
        return torch.cat([pos.detach(), centroid.expand(N - n, -1)], 0)

    pos1 = _pad_pos(g1.pos, n1)
    pos2 = _pad_pos(g2.pos, n2)

    # --- pad adjacency ---
    def _pad_adj(adj, n):
        a = torch.zeros(N, N, device=device)
        a[:n, :n] = adj.detach()
        return a

    adj1 = _pad_adj(g1.adj, n1)
    adj2 = _pad_adj(g2.adj, n2)

    # --- greedy nearest-neighbour matching ---
    # Centre for fair distance comparison, then match
    c1, c2 = pos1.mean(0), pos2.mean(0)
    dists = torch.cdist(
        (pos1 - c1).unsqueeze(0),
        (pos2 - c2).unsqueeze(0),
    ).squeeze(0)                                       # (N, N)

    perm = torch.zeros(N, dtype=torch.long, device=device)
    used = set()
    for i in range(N):
        row = dists[i].clone()
        for u in used:
            row[u] = float('inf')
        j = int(row.argmin().item())
        perm[i] = j
        used.add(j)

    # Reorder mesh-2 so node indices correspond to mesh-1
    pos2 = pos2[perm]
    adj2 = adj2[perm][:, perm]

    # --- classify edges ---
    src_set: set[tuple[int, int]] = set()
    tgt_set: set[tuple[int, int]] = set()
    for i in range(N):
        for j in range(i + 1, N):
            if adj1[i, j].item() > 0.5:
                src_set.add((i, j))
            if adj2[i, j].item() > 0.5:
                tgt_set.add((i, j))

    shared   = src_set & tgt_set
    only_src = sorted(src_set - shared)
    only_tgt = sorted(tgt_set - shared)

    # --- generate frames ---
    graphs: list[SpatialGraph] = []
    for s in range(steps + 1):
        t = s / steps

        # Linear position interpolation
        pos_t = (1.0 - t) * pos1 + t * pos2

        # Gradual edge transition
        n_src_keep = int(round(len(only_src) * (1.0 - t)))
        n_tgt_add  = int(round(len(only_tgt) * t))

        active = set(shared)
        active.update(only_src[:n_src_keep])
        active.update(only_tgt[:n_tgt_add])

        adj_t = torch.zeros(N, N, device=device)
        for (i, j) in active:
            adj_t[i, j] = 1.0
            adj_t[j, i] = 1.0

        parent = torch.full((N,), -1, dtype=torch.long, device=device)
        graphs.append(SpatialGraph(pos=pos_t, adj=adj_t, parent=parent))

    return graphs


def showcase_meshes(device: torch.device | None = None) -> list[tuple[str, SpatialGraph]]:
    """Return the six canonical showcase shapes with their names.

    These are the fixed procedural meshes displayed in the UI grid
    and available for latent-space interpolation.
    """
    return [
        ("Cube",                cube(device=device)),
        ("Octahedron",          octahedron(device=device)),
        ("Deformed Icosahedron", deformed_icosahedron(device=device)),
        ("Hexagonal Prism",     hexagonal_prism(device=device)),
        ("3D Star",             star_3d(device=device)),
        ("Low-Poly Torus",      low_poly_torus(device=device)),
    ]


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
    """Generate a diverse dataset of synthetic meshes for training.

    Cycles through all six shape generators (cube, octahedron,
    icosahedron, hexagonal prism, 3D star, torus) with randomised
    scale and noise so the VAE learns to reconstruct every shape type.

    Args:
        num: number of meshes to generate
        mesh_type: ignored (kept for API compatibility)
        num_points_range: ignored (kept for API compatibility)
        device: torch device
    """

    def _rscale():
        return 0.8 + torch.rand(1).item() * 0.4

    generators = [
        lambda: cube(scale=_rscale(), noise=0.06 + torch.rand(1).item() * 0.10, device=device),
        lambda: octahedron(scale=_rscale(), noise=0.05 + torch.rand(1).item() * 0.08, device=device),
        lambda: deformed_icosahedron(scale=_rscale(), noise=0.10 + torch.rand(1).item() * 0.15, device=device),
        lambda: hexagonal_prism(scale=_rscale(), noise=0.05 + torch.rand(1).item() * 0.08, device=device),
        lambda: star_3d(scale=_rscale(), noise=0.03 + torch.rand(1).item() * 0.06, device=device),
        lambda: low_poly_torus(
            minor_r=0.25 + torch.rand(1).item() * 0.20,
            noise=0.03 + torch.rand(1).item() * 0.05,
            device=device,
        ),
    ]

    graphs = []
    for i in range(num):
        gen = generators[i % len(generators)]
        graphs.append(gen())
    return graphs
