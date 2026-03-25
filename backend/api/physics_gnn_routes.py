"""API routes for physics-informed graph neural networks.

Provides interactive demos for:
    - Synthetic spatial graph datasets with ground-truth physics labels
    - Training PhysicsInformedGNN with live loss tracking
    - Ablation studies showing contribution of each physics component
    - Reaction-diffusion pattern generation on graphs
    - Curvature and energy analysis of spatial graphs
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import math
import time
from typing import Optional

from backend.core.physics_gnn import (
    PhysicsInformedGNN,
    PhysicsInformedGraphGenerator,
    TrainConfig,
    train_node_model,
    run_ablation,
    print_ablation_table,
    AblationResult,
    discrete_curvatures,
    geometric_edge_weights,
    weighted_laplacian,
    heat_kernel,
    multiscale_diffusion_filters,
    dirichlet_energy_from_positions,
    total_variation,
    elastic_energy,
)
from backend.config import DEVICE

router = APIRouter(prefix="/physics", tags=["physics-gnn"])

_state = {
    'model': None,
    'generator': None,
    'train_data': None,
    'ablation_results': None,
    'last_curvatures': None,
}


# ---------------------------------------------------------------------------
# Synthetic data generators — physical systems as graph problems
# ---------------------------------------------------------------------------

def _make_grid_graph(n_side: int = 8, noise: float = 0.1):
    """2D grid with optional positional noise — models a sensor array or lattice."""
    N = n_side * n_side
    pos = torch.zeros(N, 3)
    adj = torch.zeros(N, N)

    for r in range(n_side):
        for c in range(n_side):
            idx = r * n_side + c
            pos[idx] = torch.tensor([float(c), float(r), 0.0])
            if c + 1 < n_side:
                adj[idx, idx + 1] = 1
                adj[idx + 1, idx] = 1
            if r + 1 < n_side:
                adj[idx, idx + n_side] = 1
                adj[idx + n_side, idx] = 1

    if noise > 0:
        pos = pos + torch.randn_like(pos) * noise

    return pos, adj, N


def _make_molecular_graph(num_nodes: int = 30):
    """Random 3D spatial graph resembling a molecular structure."""
    pos = torch.randn(num_nodes, 3) * 2.0

    # Connect nearby atoms (distance-based adjacency)
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = diff.norm(dim=2)
    threshold = dist[dist > 0].quantile(0.25).item()
    adj = ((dist < threshold) & (dist > 0)).float()

    # Ensure connected: add edges to isolated nodes
    degree = adj.sum(dim=1)
    for i in range(num_nodes):
        if degree[i] < 1:
            dists = dist[i].clone()
            dists[i] = float('inf')
            nearest = dists.argmin()
            adj[i, nearest] = 1
            adj[nearest, i] = 1

    return pos, adj


def _generate_heat_labels(pos, adj, n_sources=3):
    """Heat diffusion from random sources — smooth, physically motivated labels."""
    N = pos.size(0)
    W = geometric_edge_weights(pos, adj)
    L = weighted_laplacian(W)
    K = heat_kernel(L, t=2.0)

    # Place heat sources
    sources = torch.zeros(N)
    src_idx = torch.randperm(N)[:n_sources]
    sources[src_idx] = 1.0

    # Diffuse
    temperature = K @ sources
    temperature = (temperature - temperature.min()) / (temperature.max() - temperature.min() + 1e-8)

    # Discretize to 3 classes: cold, warm, hot
    labels = torch.zeros(N, dtype=torch.long)
    labels[temperature > 0.33] = 1
    labels[temperature > 0.66] = 2

    return labels, temperature


def _generate_curvature_labels(pos, adj):
    """Label nodes by local geometric curvature — flat, curved, highly curved."""
    curv = discrete_curvatures(pos, adj)
    curvedness = curv['curvedness']

    # Normalize
    c_norm = (curvedness - curvedness.min()) / (curvedness.max() - curvedness.min() + 1e-8)

    labels = torch.zeros(pos.size(0), dtype=torch.long)
    labels[c_norm > 0.33] = 1
    labels[c_norm > 0.66] = 2

    return labels, curv


def _generate_stress_labels(pos, adj):
    """Elastic stress classification — nodes under tension vs compression."""
    edges = adj.triu().nonzero(as_tuple=False)
    if edges.size(0) == 0:
        return torch.zeros(pos.size(0), dtype=torch.long), torch.zeros(pos.size(0))

    edge_lens = (pos[edges[:, 0]] - pos[edges[:, 1]]).norm(dim=1)
    mean_len = edge_lens.mean()

    # Per-node stress: average deviation from mean edge length
    N = pos.size(0)
    stress = torch.zeros(N)
    count = torch.zeros(N)
    for k in range(edges.size(0)):
        i, j = edges[k, 0].item(), edges[k, 1].item()
        s = (edge_lens[k] - mean_len).abs()
        stress[i] += s
        stress[j] += s
        count[i] += 1
        count[j] += 1
    stress = stress / count.clamp(min=1)

    s_norm = (stress - stress.min()) / (stress.max() - stress.min() + 1e-8)
    labels = torch.zeros(N, dtype=torch.long)
    labels[s_norm > 0.33] = 1
    labels[s_norm > 0.66] = 2

    return labels, stress


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_graph(pos, adj, labels=None, values=None, curvatures=None, name=None):
    """Convert graph tensors to JSON-serializable dict."""
    N = pos.size(0)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j].item() > 0.5:
                edges.append([i, j])

    result = {
        'num_nodes': N,
        'num_edges': len(edges),
        'positions': pos.detach().cpu().tolist(),
        'edges': edges,
        'type': 'physics',
    }

    if labels is not None:
        result['labels'] = labels.detach().cpu().tolist()
    if values is not None:
        result['values'] = values.detach().cpu().tolist()
    if curvatures is not None:
        result['curvatures'] = {
            k: v.detach().cpu().tolist() for k, v in curvatures.items()
        }
    if name is not None:
        result['name'] = name

    return result


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

class DatasetRequest(BaseModel):
    scenario: str = 'heat_diffusion'
    num_nodes: int = 40
    grid_size: int = 7
    noise: float = 0.1


@router.post("/dataset")
def generate_dataset(req: DatasetRequest):
    """Generate a synthetic spatial graph with physics-based node labels.

    Scenarios:
        heat_diffusion  — classify nodes by diffused temperature
        curvature       — classify nodes by local geometric curvature
        stress          — classify nodes by elastic stress
        molecular       — 3D molecular graph with curvature labels
    """
    if req.scenario == 'heat_diffusion':
        pos, adj, N = _make_grid_graph(req.grid_size, req.noise)
        labels, values = _generate_heat_labels(pos, adj)
        curv = discrete_curvatures(pos, adj)
        graph = _serialize_graph(pos, adj, labels, values, curv, 'Heat Diffusion on Lattice')

    elif req.scenario == 'curvature':
        pos, adj = _make_molecular_graph(req.num_nodes)
        labels, curv = _generate_curvature_labels(pos, adj)
        values = curv['curvedness']
        graph = _serialize_graph(pos, adj, labels, values, curv, 'Curvature Classification')

    elif req.scenario == 'stress':
        pos, adj = _make_molecular_graph(req.num_nodes)
        labels, values = _generate_stress_labels(pos, adj)
        curv = discrete_curvatures(pos, adj)
        graph = _serialize_graph(pos, adj, labels, values, curv, 'Elastic Stress Analysis')

    elif req.scenario == 'molecular':
        pos, adj = _make_molecular_graph(req.num_nodes)
        labels, curv = _generate_curvature_labels(pos, adj)
        graph = _serialize_graph(pos, adj, labels, curv['curvedness'], curv, 'Molecular Structure')

    else:
        raise HTTPException(400, f"Unknown scenario: {req.scenario}")

    # Store for training
    _state['train_data'] = {
        'pos': pos.to(DEVICE),
        'adj': adj.to(DEVICE),
        'labels': labels.to(DEVICE),
        'scenario': req.scenario,
    }

    return {'graph': graph, 'scenario': req.scenario, 'num_classes': 3}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    hidden_dim: int = 32
    num_layers: int = 3
    epochs: int = 60
    lr: float = 0.001
    physics_weight: float = 0.01
    use_curvature: bool = True
    use_diffusion: bool = True
    use_reaction_diffusion: bool = True
    use_attention: bool = True
    regularise: bool = True
    train_fraction: float = 0.6


@router.post("/train")
def train_model(req: TrainRequest):
    """Train a PhysicsInformedGNN on the current dataset."""
    if _state['train_data'] is None:
        raise HTTPException(400, "No dataset loaded. Call /physics/dataset first.")

    data = _state['train_data']
    pos = data['pos']
    adj = data['adj']
    labels = data['labels']
    N = pos.size(0)

    # Train/val split
    perm = torch.randperm(N, device=DEVICE)
    n_train = int(N * req.train_fraction)
    train_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:]] = True

    model = PhysicsInformedGNN(
        in_channels=0,
        hidden_channels=req.hidden_dim,
        out_channels=3,
        num_layers=req.num_layers,
        task='node',
        use_curvature=req.use_curvature,
        use_diffusion=req.use_diffusion,
        use_reaction_diffusion=req.use_reaction_diffusion,
        use_attention=req.use_attention,
        regularise=req.regularise,
        dropout=0.1,
    ).to(DEVICE)

    config = TrainConfig(
        epochs=req.epochs,
        lr=req.lr,
        physics_weight=req.physics_weight,
        patience=max(req.epochs, 30),
        log_interval=req.epochs + 1,
    )

    t0 = time.time()
    history = train_node_model(
        model, pos, adj, labels, train_mask, val_mask,
        config=config, task_loss='cross_entropy',
    )
    train_time = time.time() - t0

    _state['model'] = model
    _state['train_mask'] = train_mask
    _state['val_mask'] = val_mask

    # Evaluate
    model.eval()
    with torch.no_grad():
        out = model(pos, adj)
        pred = out.argmax(dim=1)
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean().item()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean().item()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'train_losses': [round(l, 4) for l in history['train_losses']],
        'val_losses': [round(l, 4) for l in history['val_losses']],
        'train_accuracy': round(train_acc, 4),
        'val_accuracy': round(val_acc, 4),
        'best_epoch': history['best_epoch'],
        'train_time': round(train_time, 2),
        'num_params': n_params,
        'predictions': pred.detach().cpu().tolist(),
    }


# ---------------------------------------------------------------------------
# Prediction / inference
# ---------------------------------------------------------------------------

@router.post("/predict")
def predict():
    """Run inference with the trained model, return per-node predictions."""
    if _state['model'] is None:
        raise HTTPException(400, "No trained model. Call /physics/train first.")
    if _state['train_data'] is None:
        raise HTTPException(400, "No dataset loaded.")

    data = _state['train_data']
    model = _state['model']
    model.eval()

    with torch.no_grad():
        out, energy = model(data['pos'], data['adj'], return_energy=True)
        pred = out.argmax(dim=1)
        probs = F.softmax(out, dim=1)

    return {
        'predictions': pred.detach().cpu().tolist(),
        'probabilities': probs.detach().cpu().tolist(),
        'energy': round(energy.item(), 4),
    }


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

class AblationRequest(BaseModel):
    epochs: int = 40
    hidden_dim: int = 32


@router.post("/ablation")
def run_ablation_study(req: AblationRequest):
    """Run ablation comparing all physics components."""
    if _state['train_data'] is None:
        raise HTTPException(400, "No dataset loaded. Call /physics/dataset first.")

    data = _state['train_data']
    pos = data['pos']
    adj = data['adj']
    labels = data['labels']
    N = pos.size(0)

    perm = torch.randperm(N, device=DEVICE)
    n_train = int(N * 0.6)
    train_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:]] = True

    results = run_ablation(
        pos, adj, labels, train_mask, val_mask,
        task_loss='cross_entropy',
        epochs=req.epochs,
    )

    _state['ablation_results'] = results

    return {
        'results': [
            {
                'name': r.name,
                'accuracy': round(r.val_metric, 4),
                'train_loss': round(r.train_loss, 4),
                'val_loss': round(r.val_loss, 4),
                'num_params': r.num_params,
                'train_time': round(r.train_time, 2),
            }
            for r in results
        ]
    }


# ---------------------------------------------------------------------------
# Energy analysis
# ---------------------------------------------------------------------------

@router.post("/energy")
def compute_energy():
    """Compute physics energy breakdown for the current graph."""
    if _state['train_data'] is None:
        raise HTTPException(400, "No dataset loaded.")

    data = _state['train_data']
    pos = data['pos']
    adj = data['adj']

    # Use model predictions if available, else random signal
    if _state['model'] is not None:
        _state['model'].eval()
        with torch.no_grad():
            f = _state['model'](pos, adj)
    else:
        f = torch.randn(pos.size(0), 4, device=DEVICE)

    E_dirichlet = dirichlet_energy_from_positions(f, pos, adj).item()
    E_tv = total_variation(f, pos, adj, p=1.0).item()
    E_elastic = elastic_energy(pos, adj).item()

    return {
        'dirichlet': round(E_dirichlet, 4),
        'total_variation': round(E_tv, 4),
        'elastic': round(E_elastic, 4),
        'total': round(E_dirichlet + E_tv + E_elastic, 4),
    }


# ---------------------------------------------------------------------------
# Curvature analysis
# ---------------------------------------------------------------------------

@router.post("/curvatures")
def compute_curvatures():
    """Compute discrete curvature quantities for the current graph."""
    if _state['train_data'] is None:
        raise HTTPException(400, "No dataset loaded.")

    data = _state['train_data']
    curv = discrete_curvatures(data['pos'], data['adj'])

    result = {}
    for key in ['mean', 'gaussian', 'principal_1', 'principal_2', 'shape_index', 'curvedness']:
        vals = curv[key].detach().cpu()
        result[key] = {
            'values': vals.tolist(),
            'min': round(vals.min().item(), 4),
            'max': round(vals.max().item(), 4),
            'mean': round(vals.mean().item(), 4),
            'std': round(vals.std().item(), 4),
        }

    _state['last_curvatures'] = curv
    return result


# ---------------------------------------------------------------------------
# Heat diffusion visualization
# ---------------------------------------------------------------------------

class HeatDiffusionRequest(BaseModel):
    source_nodes: list[int] = [0]
    num_steps: int = 5
    t_max: float = 5.0


@router.post("/heat_diffusion")
def heat_diffusion_viz(req: HeatDiffusionRequest):
    """Visualize heat diffusion from source nodes at multiple time steps."""
    if _state['train_data'] is None:
        raise HTTPException(400, "No dataset loaded.")

    data = _state['train_data']
    pos = data['pos']
    adj = data['adj']
    N = pos.size(0)

    W = geometric_edge_weights(pos, adj)
    L = weighted_laplacian(W)

    # Initial signal: delta at sources
    sources = torch.zeros(N, device=DEVICE)
    for s in req.source_nodes:
        if 0 <= s < N:
            sources[s] = 1.0

    times = [req.t_max * i / max(req.num_steps - 1, 1) for i in range(req.num_steps)]
    times[0] = 0.01  # avoid t=0

    steps = []
    for t in times:
        K = heat_kernel(L, t)
        u = K @ sources
        u_norm = (u - u.min()) / (u.max() - u.min() + 1e-8)
        steps.append({
            't': round(t, 3),
            'values': u_norm.detach().cpu().tolist(),
        })

    return {'steps': steps, 'source_nodes': req.source_nodes}


# ---------------------------------------------------------------------------
# Graph generation via reaction-diffusion
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    num_samples: int = 4
    latent_dim: int = 16
    max_nodes: int = 20
    num_rd_steps: int = 4


@router.post("/generate")
def generate_graphs(req: GenerateRequest):
    """Generate spatial graphs using reaction-diffusion morphogenesis."""
    gen = PhysicsInformedGraphGenerator(
        latent_dim=req.latent_dim,
        hidden_channels=32,
        max_nodes=req.max_nodes,
        num_rd_steps=req.num_rd_steps,
        num_species=2,
    ).to(DEVICE)

    samples = gen.generate(num_samples=req.num_samples, device=DEVICE, threshold=0.4)

    results = []
    for s in samples:
        pos = s['positions']
        adj_mat = s['adjacency']
        n = s['num_nodes']

        if n < 2:
            continue

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_mat[i, j].item() > 0.5:
                    edges.append([i, j])

        results.append({
            'num_nodes': n,
            'num_edges': len(edges),
            'positions': pos.detach().cpu().tolist(),
            'edges': edges,
            'type': 'generated',
            'name': f'RD Sample ({n} nodes)',
        })

    return {'graphs': results}


# ---------------------------------------------------------------------------
# Reaction-diffusion pattern visualization
# ---------------------------------------------------------------------------

class RDPatternRequest(BaseModel):
    grid_size: int = 10
    feed_rate: float = 0.055
    kill_rate: float = 0.062
    num_steps: int = 8
    steps_per_frame: int = 50


@router.post("/rd_pattern")
def reaction_diffusion_pattern(req: RDPatternRequest):
    """Run Gray-Scott reaction-diffusion on a grid and return snapshots."""
    pos, adj, N = _make_grid_graph(req.grid_size, noise=0.0)
    pos = pos.to(DEVICE)
    adj = adj.to(DEVICE)

    W = geometric_edge_weights(pos, adj)
    L = weighted_laplacian(W)

    # Normalize Laplacian
    D = (-L.diagonal()).clamp(min=1e-8)
    D_inv_sqrt = D.pow(-0.5)
    L_norm = D_inv_sqrt.unsqueeze(1) * L * D_inv_sqrt.unsqueeze(0)

    # Initialize concentrations
    A = torch.ones(N, device=DEVICE)
    B = torch.zeros(N, device=DEVICE)

    # Seed B in a small region near the center
    center = N // 2
    neighbors = (adj[center] > 0).nonzero(as_tuple=True)[0]
    seed_nodes = torch.cat([torch.tensor([center], device=DEVICE), neighbors])
    B[seed_nodes] = 1.0

    Da, Db = 0.16, 0.08
    f, k = req.feed_rate, req.kill_rate
    dt = 0.5

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j].item() > 0.5:
                edges.append([i, j])

    frames = [{
        'step': 0,
        'A': A.detach().cpu().tolist(),
        'B': B.detach().cpu().tolist(),
    }]

    for frame_idx in range(1, req.num_steps):
        for _ in range(req.steps_per_frame):
            reaction = A * B * B
            A_new = A + dt * (Da * (L_norm @ A) - reaction + f * (1 - A))
            B_new = B + dt * (Db * (L_norm @ B) + reaction - (f + k) * B)
            A = A_new.clamp(0, 1)
            B = B_new.clamp(0, 1)

        frames.append({
            'step': frame_idx * req.steps_per_frame,
            'A': A.detach().cpu().tolist(),
            'B': B.detach().cpu().tolist(),
        })

    return {
        'frames': frames,
        'positions': pos.detach().cpu().tolist(),
        'edges': edges,
        'num_nodes': N,
        'grid_size': req.grid_size,
        'params': {'feed': f, 'kill': k, 'Da': Da, 'Db': Db},
    }
