"""API routes for hyperbolic GNN demos.

Provides interactive demos for:
    - Graph generation for hyperbolic embedding (trees, karate club, hierarchical, grid)
    - Spring-mass simulation on the Poincare disk
    - Hyperbolic GNN training (GCN / GAT layers)
    - Euclidean vs hyperbolic embedding comparison
    - Geodesic arc computation on the Poincare disk
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import math
from typing import Optional

from backend.core.hyperbolic import (
    PoincareBall,
    HyperbolicGNN,
    HyperbolicSimulation,
    HyperbolicTrainConfig,
    train_hyperbolic_gnn,
    compare_embeddings,
    compute_geodesic_arc,
)

router = APIRouter(prefix="/hyperbolic", tags=["hyperbolic"])

_state = {
    'positions': None,   # tensor (N, 2) Poincare disk positions
    'edges': None,       # list of [i, j]
    'labels': None,      # tensor (N,)
    'num_nodes': 0,
    'simulation': None,  # HyperbolicSimulation instance
    'model': None,       # trained HyperbolicGNN
    'features': None,    # node features tensor (N, F)
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _make_binary_tree(depth: int):
    """Generate a full binary tree with the given depth (number of levels).

    Returns:
        edges: list of [parent, child] pairs.
        num_nodes: total number of nodes.
        labels: list of int labels (depth of each node).
    """
    num_nodes = (1 << depth) - 1  # 2^depth - 1
    edges = []
    labels = [0] * num_nodes

    for i in range(num_nodes):
        left = 2 * i + 1
        right = 2 * i + 2
        # Compute depth of node i
        node_depth = int(math.log2(i + 1))
        labels[i] = node_depth
        if left < num_nodes:
            edges.append([i, left])
        if right < num_nodes:
            edges.append([i, right])

    return edges, num_nodes, labels


def _make_hierarchical_graph(num_nodes: int):
    """Generate a random hierarchical graph with community structure.

    Creates 3-4 communities connected in a tree-like hierarchy.

    Returns:
        edges: list of [i, j] pairs.
        labels: list of int community labels.
    """
    n_communities = min(4, max(2, num_nodes // 8))
    community_size = num_nodes // n_communities
    edges = []
    labels = [0] * num_nodes

    # Assign nodes to communities
    for i in range(num_nodes):
        labels[i] = min(i // community_size, n_communities - 1)

    # Intra-community edges (denser)
    for c in range(n_communities):
        start = c * community_size
        end = start + community_size if c < n_communities - 1 else num_nodes
        nodes = list(range(start, end))
        # Create a random tree within community
        for i in range(1, len(nodes)):
            parent = nodes[torch.randint(0, i, (1,)).item()]
            edges.append([parent, nodes[i]])
        # Add extra intra-community edges
        n_extra = max(1, len(nodes) // 3)
        for _ in range(n_extra):
            a = nodes[torch.randint(0, len(nodes), (1,)).item()]
            b = nodes[torch.randint(0, len(nodes), (1,)).item()]
            if a != b:
                edges.append([a, b])

    # Inter-community edges (sparser, tree hierarchy)
    for c in range(1, n_communities):
        parent_comm = (c - 1) // 2
        src_start = parent_comm * community_size
        dst_start = c * community_size
        src = src_start + torch.randint(0, community_size, (1,)).item()
        dst = dst_start + torch.randint(0, community_size, (1,)).item()
        edges.append([src, dst])

    return edges, labels


def _make_grid_graph(size: int):
    """Generate a 2D grid graph of size x size.

    Returns:
        edges: list of [i, j] pairs.
        num_nodes: total number of nodes.
        labels: list of int labels (quadrant-based).
    """
    num_nodes = size * size
    edges = []
    labels = [0] * num_nodes

    for r in range(size):
        for c in range(size):
            idx = r * size + c
            # Label by quadrant
            qr = 0 if r < size // 2 else 1
            qc = 0 if c < size // 2 else 1
            labels[idx] = qr * 2 + qc
            if c + 1 < size:
                edges.append([idx, idx + 1])
            if r + 1 < size:
                edges.append([idx, idx + size])

    return edges, num_nodes, labels


def _init_poincare_positions(num_nodes: int, c: float = 1.0):
    """Initialize random positions inside the Poincare disk.

    Uses expmap0 from small random tangent vectors to ensure
    positions lie within the disk (radius < 0.9).

    Returns:
        positions: (num_nodes, 2) tensor on the Poincare disk.
    """
    manifold = PoincareBall(c=c)
    # Small random tangent vectors at the origin
    tangent = 0.3 * torch.randn(num_nodes, 2)
    positions = manifold.expmap0(tangent)
    # Clamp to keep away from boundary
    norms = positions.norm(dim=1, keepdim=True)
    scale = torch.clamp(norms, max=0.85) / norms.clamp(min=1e-8)
    positions = positions * scale
    return positions


def _serialize_positions(tensor):
    """Convert a 2D tensor to a list of [x, y] pairs."""
    return tensor.detach().cpu().tolist()


def _make_karate_club():
    """Generate Zachary's karate club graph.

    Returns:
        edges: list of [i, j] pairs.
        num_nodes: 34.
        labels: list of int community labels (2 communities).
    """
    # Zachary's karate club edges (0-indexed)
    edge_list = [
        (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),
        (0,12),(0,13),(0,17),(0,19),(0,21),(0,31),
        (1,2),(1,3),(1,7),(1,13),(1,17),(1,19),(1,21),(1,30),
        (2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),(2,32),
        (3,7),(3,12),(3,13),
        (4,6),(4,10),
        (5,6),(5,10),(5,16),
        (6,16),
        (8,30),(8,32),(8,33),
        (9,33),
        (13,33),
        (14,32),(14,33),
        (15,32),(15,33),
        (18,32),(18,33),
        (19,33),
        (20,32),(20,33),
        (22,32),(22,33),
        (23,25),(23,27),(23,29),(23,32),(23,33),
        (24,25),(24,27),(24,31),
        (25,31),
        (26,29),(26,33),
        (27,33),
        (28,31),(28,33),
        (29,32),(29,33),
        (30,32),(30,33),
        (31,32),(31,33),
        (32,33),
    ]
    edges = [[i, j] for i, j in edge_list]
    num_nodes = 34

    # Ground truth communities (Mr. Hi=0 vs Officer=1)
    community_0 = {0,1,2,3,4,5,6,7,8,10,11,12,13,16,17,19,21}
    labels = [0 if i in community_0 else 1 for i in range(num_nodes)]

    return edges, num_nodes, labels


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class GraphRequest(BaseModel):
    graph_type: str = "binary_tree"
    num_nodes: int = 4  # depth for binary_tree, num_nodes for hierarchical, size for grid
    branching_factor: int = 2


@router.post("/graph")
def generate_graph(req: GraphRequest):
    """Generate a graph for hyperbolic embedding.

    Graph types: binary_tree, karate_club, random_hierarchical, grid.
    """
    if req.graph_type == "binary_tree":
        edges, num_nodes, labels = _make_binary_tree(req.num_nodes)
    elif req.graph_type == "karate_club":
        edges, num_nodes, labels = _make_karate_club()
    elif req.graph_type == "random_hierarchical":
        n = max(8, req.num_nodes)
        edges, labels = _make_hierarchical_graph(n)
        num_nodes = n
    elif req.graph_type == "grid":
        size = max(2, req.num_nodes)
        edges, num_nodes, labels = _make_grid_graph(size)
    else:
        raise HTTPException(400, f"Unknown graph_type: {req.graph_type}")

    # Initialize Poincare disk positions
    positions = _init_poincare_positions(num_nodes)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Build adjacency matrix for later use
    adj = torch.zeros(num_nodes, num_nodes)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # Generate simple node features (one-hot degree + position)
    degree = adj.sum(dim=1, keepdim=True)
    features = torch.cat([degree, positions], dim=1)  # (N, 3)

    # Store state
    _state['positions'] = positions
    _state['edges'] = edges
    _state['labels'] = labels_tensor
    _state['num_nodes'] = num_nodes
    _state['features'] = features

    # Build response
    nodes = []
    for idx in range(num_nodes):
        nodes.append({
            'id': idx,
            'x': positions[idx, 0].item(),
            'y': positions[idx, 1].item(),
            'label': labels[idx],
        })

    return {
        'nodes': nodes,
        'edges': edges,
        'num_nodes': num_nodes,
        'num_edges': len(edges),
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class SimInitRequest(BaseModel):
    curvature: float = 1.0
    spring_k: float = 2.0
    target_length: float = 0.5
    charge_c: float = 0.1
    lr: float = 0.03


@router.post("/simulate/init")
def simulate_init(req: SimInitRequest):
    """Initialize spring-mass simulation on the Poincare disk."""
    if _state['positions'] is None or _state['edges'] is None:
        raise HTTPException(400, "No graph loaded. Call /hyperbolic/graph first.")

    params = {
        'curvature': req.curvature,
        'spring_k': req.spring_k,
        'target_L': req.target_length,
        'charge_c': req.charge_c,
        'lr': req.lr,
    }

    sim = HyperbolicSimulation(
        n_nodes=_state['num_nodes'],
        edges=[(e[0], e[1]) for e in _state['edges']],
        params=params,
        initial_positions=_state['positions'].clone(),
    )

    _state['simulation'] = sim

    pos = sim.get_positions()
    return {
        'positions': _serialize_positions(pos),
        'edges': _state['edges'],
        'energy': 0.0,
    }


class SimStepRequest(BaseModel):
    n_steps: int = 10


@router.post("/simulate/step")
def simulate_step(req: SimStepRequest):
    """Run N simulation steps."""
    if _state['simulation'] is None:
        raise HTTPException(400, "No simulation initialized. Call /hyperbolic/simulate/init first.")

    sim = _state['simulation']
    pos, energy = sim.step(n_steps=req.n_steps)

    # Update stored positions
    _state['positions'] = pos.clone()

    return {
        'positions': _serialize_positions(pos),
        'energy': round(energy, 6),
        'step': req.n_steps,
    }


@router.post("/simulate/reset")
def simulate_reset():
    """Reset simulation to initial positions."""
    if _state['simulation'] is None:
        raise HTTPException(400, "No simulation initialized.")

    _state['simulation'].reset(initial_positions=_state['positions'])

    pos = _state['simulation'].get_positions()
    return {
        'positions': _serialize_positions(pos),
        'energy': 0.0,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    hidden_dim: int = 32
    num_layers: int = 2
    epochs: int = 100
    lr: float = 0.01
    curvature: float = 1.0
    layer_type: str = "gcn"  # "gcn" or "gat"


@router.post("/train")
def train_model(req: TrainRequest):
    """Train a hyperbolic GNN for node classification."""
    if _state['positions'] is None or _state['edges'] is None:
        raise HTTPException(400, "No graph loaded. Call /hyperbolic/graph first.")

    try:
        num_nodes = _state['num_nodes']
        features = _state['features']
        labels = _state['labels']

        # Build adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes)
        for i, j in _state['edges']:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        num_classes = int(labels.max().item()) + 1
        use_attention = (req.layer_type == "gat")

        model = HyperbolicGNN(
            in_channels=features.size(1),
            hidden_channels=req.hidden_dim,
            out_channels=num_classes,
            num_layers=req.num_layers,
            c=req.curvature,
            dropout=0.1,
            use_attention=use_attention,
        )

        # Train/val split
        perm = torch.randperm(num_nodes)
        n_train = int(num_nodes * 0.6)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:]] = True

        config = HyperbolicTrainConfig(
            epochs=req.epochs,
            lr=req.lr,
            patience=max(req.epochs, 30),
            log_interval=req.epochs + 1,  # suppress printing
        )

        history = train_hyperbolic_gnn(
            model=model,
            features=features,
            adj=adj,
            labels=labels,
            config=config,
            train_mask=train_mask,
            val_mask=val_mask,
        )

        _state['model'] = model

        # Final evaluation
        model.eval()
        with torch.no_grad():
            logits = model(features, adj)
            pred = logits.argmax(dim=1)
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean().item()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean().item()
            embeddings = model.get_embeddings(features, adj)

        # Project embeddings to 2D for visualization if needed
        emb_2d = embeddings[:, :2] if embeddings.size(1) >= 2 else embeddings

        return {
            'losses': [round(l, 4) for l in history['losses']],
            'train_acc': round(train_acc, 4),
            'val_acc': round(val_acc, 4),
            'embeddings': _serialize_positions(emb_2d),
            'predictions': pred.detach().cpu().tolist(),
        }

    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    embed_dim: int = 16
    epochs: int = 300
    lr: float = 0.01


@router.post("/compare")
def compare_embeddings_endpoint(req: CompareRequest):
    """Compare Euclidean vs hyperbolic embeddings on the current graph."""
    if _state['edges'] is None or _state['num_nodes'] == 0:
        raise HTTPException(400, "No graph loaded. Call /hyperbolic/graph first.")

    try:
        num_nodes = _state['num_nodes']
        edge_tensor = torch.tensor(_state['edges'], dtype=torch.long)

        result = compare_embeddings(
            num_nodes=num_nodes,
            edges=edge_tensor,
            embed_dim=req.embed_dim,
            epochs=req.epochs,
            lr=req.lr,
        )

        euc = result['euclidean']
        hyp = result['poincare']

        # Project to 2D for visualization
        euc_pos = euc['embeddings'][:, :2] if euc['embeddings'].size(1) >= 2 else euc['embeddings']
        hyp_pos = hyp['embeddings'][:, :2] if hyp['embeddings'].size(1) >= 2 else hyp['embeddings']

        return {
            'euclidean': {
                'positions': _serialize_positions(euc_pos),
                'distortion': round(euc['distortion'], 6),
                'avg_dist_error': round(euc['mean_edge_dist'], 6),
            },
            'hyperbolic': {
                'positions': _serialize_positions(hyp_pos),
                'distortion': round(hyp['distortion'], 6),
                'avg_dist_error': round(hyp['mean_edge_dist'], 6),
            },
        }

    except Exception as e:
        raise HTTPException(500, f"Comparison failed: {str(e)}")


# ---------------------------------------------------------------------------
# Geodesics
# ---------------------------------------------------------------------------

class GeodesicRequest(BaseModel):
    n_samples: int = 20


@router.post("/geodesics")
def compute_geodesics(req: GeodesicRequest):
    """Compute geodesic arcs for all edges of the current graph."""
    if _state['positions'] is None or _state['edges'] is None:
        raise HTTPException(400, "No graph loaded. Call /hyperbolic/graph first.")

    positions = _state['positions']
    arcs = []

    for edge in _state['edges']:
        i, j = edge[0], edge[1]
        p1 = positions[i]
        p2 = positions[j]
        arc = compute_geodesic_arc(p1, p2, n_samples=req.n_samples)
        arcs.append(arc.detach().cpu().tolist())

    return {'arcs': arcs}
