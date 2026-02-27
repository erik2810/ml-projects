"""
Spatial graph data structure for trees embedded in 3D space.

Core abstraction: a rooted tree where each node carries a 3D position
and optional features. Edges encode parent-child relationships —
crucially, connectivity is NOT implied by spatial proximity (unlike
molecules). This distinction is fundamental for neuronal/botanical
morphologies.

SWC is the de-facto standard for neuron reconstructions (Cannon et al., 1998).
We parse it into SpatialGraph and can export back.

SWC columns: id  type  x  y  z  radius  parent_id
"""

import torch
from torch import Tensor
from dataclasses import dataclass, field


@dataclass
class SpatialGraph:
    """A tree (or forest) embedded in R^3.

    Attributes:
        pos:        (N, 3)  node positions in 3D
        adj:        (N, N)  adjacency matrix (symmetric, unweighted)
        parent:     (N,)    parent index for each node (-1 for roots)
        radii:      (N,)    optional radius per node (e.g. dendrite thickness)
        node_types: (N,)    integer type labels (SWC convention: 1=soma, 3=basal, 4=apical, ...)
        features:   (N, F)  optional additional node features
    """
    pos: Tensor
    adj: Tensor
    parent: Tensor
    radii: Tensor | None = None
    node_types: Tensor | None = None
    features: Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return self.pos.size(0)

    @property
    def num_edges(self) -> int:
        return int(self.adj.sum().item()) // 2

    @property
    def root(self) -> int:
        roots = (self.parent == -1).nonzero(as_tuple=True)[0]
        return roots[0].item() if len(roots) > 0 else 0

    @property
    def device(self):
        return self.pos.device

    def children_of(self, node: int) -> list[int]:
        return (self.parent == node).nonzero(as_tuple=True)[0].tolist()

    def depth(self, node: int) -> int:
        """Distance from node to root (number of edges)."""
        d = 0
        cur = node
        while self.parent[cur].item() != -1:
            cur = int(self.parent[cur].item())
            d += 1
        return d

    def subtree_sizes(self) -> Tensor:
        """Compute subtree size for each node via bottom-up traversal."""
        n = self.num_nodes
        sizes = torch.ones(n, dtype=torch.long, device=self.device)
        # process leaves first (topological sort by depth)
        order = sorted(range(n), key=lambda i: self.depth(i), reverse=True)
        for i in order:
            p = int(self.parent[i].item())
            if p >= 0:
                sizes[p] += sizes[i]
        return sizes

    def segment_lengths(self) -> Tensor:
        """Euclidean length of each edge (child -> parent)."""
        mask = self.parent >= 0
        children = mask.nonzero(as_tuple=True)[0]
        parents = self.parent[children].long()
        diffs = self.pos[children] - self.pos[parents]
        return torch.norm(diffs, dim=1)

    def branch_angles(self) -> Tensor:
        """Angle between sibling pairs at each branching point (radians).

        For a node with children c1, c2 the angle is computed from
        the vectors (parent -> c1) and (parent -> c2).
        """
        angles = []
        for node in range(self.num_nodes):
            kids = self.children_of(node)
            if len(kids) < 2:
                continue
            # pairwise angles among siblings
            for i in range(len(kids)):
                for j in range(i + 1, len(kids)):
                    v1 = self.pos[kids[i]] - self.pos[node]
                    v2 = self.pos[kids[j]] - self.pos[node]
                    cos_a = torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-8)
                    angles.append(torch.acos(cos_a.clamp(-1, 1)))
        if not angles:
            return torch.tensor([], device=self.device)
        return torch.stack(angles)

    def to(self, device):
        return SpatialGraph(
            pos=self.pos.to(device),
            adj=self.adj.to(device),
            parent=self.parent.to(device),
            radii=self.radii.to(device) if self.radii is not None else None,
            node_types=self.node_types.to(device) if self.node_types is not None else None,
            features=self.features.to(device) if self.features is not None else None,
        )

    def pad(self, max_nodes: int) -> 'SpatialGraph':
        """Zero-pad to a fixed number of nodes."""
        n = self.num_nodes
        if n >= max_nodes:
            return SpatialGraph(
                pos=self.pos[:max_nodes],
                adj=self.adj[:max_nodes, :max_nodes],
                parent=self.parent[:max_nodes],
                radii=self.radii[:max_nodes] if self.radii is not None else None,
                node_types=self.node_types[:max_nodes] if self.node_types is not None else None,
            )
        dev = self.device
        pos_pad = torch.zeros(max_nodes, 3, device=dev)
        pos_pad[:n] = self.pos
        adj_pad = torch.zeros(max_nodes, max_nodes, device=dev)
        adj_pad[:n, :n] = self.adj
        parent_pad = torch.full((max_nodes,), -1, dtype=torch.long, device=dev)
        parent_pad[:n] = self.parent

        radii_pad = None
        if self.radii is not None:
            radii_pad = torch.zeros(max_nodes, device=dev)
            radii_pad[:n] = self.radii

        types_pad = None
        if self.node_types is not None:
            types_pad = torch.zeros(max_nodes, dtype=torch.long, device=dev)
            types_pad[:n] = self.node_types

        return SpatialGraph(
            pos=pos_pad, adj=adj_pad, parent=parent_pad,
            radii=radii_pad, node_types=types_pad,
        )


# ---------- SWC I/O ----------

def parse_swc(text: str, device: torch.device | None = None) -> SpatialGraph:
    """Parse SWC-format text into a SpatialGraph.

    SWC format (Cannon et al., 1998): each non-comment line is
        id  type  x  y  z  radius  parent_id
    where parent_id = -1 for the root.
    """
    lines = [l.strip() for l in text.strip().splitlines()
             if l.strip() and not l.strip().startswith('#')]

    raw = []
    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        raw.append({
            'id': int(parts[0]),
            'type': int(parts[1]),
            'x': float(parts[2]),
            'y': float(parts[3]),
            'z': float(parts[4]),
            'r': float(parts[5]),
            'parent': int(parts[6]),
        })

    if not raw:
        return SpatialGraph(
            pos=torch.zeros(0, 3), adj=torch.zeros(0, 0),
            parent=torch.tensor([]), radii=torch.tensor([]),
        )

    # remap IDs to 0-indexed (SWC IDs can start at 1 or be non-contiguous)
    id_map = {entry['id']: idx for idx, entry in enumerate(raw)}
    n = len(raw)

    pos = torch.zeros(n, 3, device=device)
    radii = torch.zeros(n, device=device)
    node_types = torch.zeros(n, dtype=torch.long, device=device)
    parent = torch.full((n,), -1, dtype=torch.long, device=device)
    adj = torch.zeros(n, n, device=device)

    for entry in raw:
        idx = id_map[entry['id']]
        pos[idx] = torch.tensor([entry['x'], entry['y'], entry['z']])
        radii[idx] = entry['r']
        node_types[idx] = entry['type']

        if entry['parent'] != -1 and entry['parent'] in id_map:
            pidx = id_map[entry['parent']]
            parent[idx] = pidx
            adj[idx, pidx] = 1.0
            adj[pidx, idx] = 1.0

    return SpatialGraph(
        pos=pos, adj=adj, parent=parent,
        radii=radii, node_types=node_types,
    )


def to_swc(graph: SpatialGraph) -> str:
    """Export a SpatialGraph to SWC format string."""
    lines = ["# SWC export from graph-ml-lab"]
    for i in range(graph.num_nodes):
        ntype = int(graph.node_types[i].item()) if graph.node_types is not None else 0
        x, y, z = graph.pos[i].tolist()
        r = float(graph.radii[i].item()) if graph.radii is not None else 1.0
        pid = int(graph.parent[i].item())
        # SWC uses 1-indexed IDs
        lines.append(f"{i+1} {ntype} {x:.4f} {y:.4f} {z:.4f} {r:.4f} {pid+1 if pid >= 0 else -1}")
    return '\n'.join(lines)
