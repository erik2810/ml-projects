"""API routes for spatial graph (3D tree) generation models.

Exposes the PhD-level spatial generation pipeline: synthetic data,
training for both VAE and diffusion approaches, generation,
evaluation, and analysis endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch

from backend.core.spatial.graph3d import SpatialGraph
from backend.core.spatial.synthetic import random_branching_tree, random_neuron_morphology
from backend.core.spatial.metrics import (
    full_evaluation, morphological_features, sholl_analysis,
)
from backend.core.spatial.tree_gen import SpatialTreeVAE, train_spatial_vae
from backend.core.spatial.diffusion3d import SpatialGraphDiffusion, train_spatial_diffusion
from backend.core.spatial.mesh_utils import (
    generate_mesh_dataset, showcase_meshes, geometric_interpolate,
)
from backend.core.spatial.mesh_vae import SpatialMeshVAE, train_mesh_vae
from backend.config import DEVICE, CHECKPOINT_DIR

router = APIRouter(prefix="/spatial", tags=["spatial"])

_state = {
    'vae_model': None,
    'diffusion_model': None,
    'train_data': None,
    'generated_vae': None,
    'generated_diff': None,
    'mesh_vae_model': None,
    'mesh_train_data': None,
    'mesh_procedural': None,
    'mesh_procedural_names': None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_graph(graph: SpatialGraph) -> dict:
    """Convert SpatialGraph to JSON-serializable dict."""
    n = graph.num_nodes
    pos = graph.pos.detach().cpu().tolist()

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if graph.adj[i, j].item() > 0.5:
                edges.append([i, j])

    parent = graph.parent.detach().cpu().tolist()

    # determine graph type: "mesh" if all parents are -1, else "tree"
    is_mesh = all(p == -1 for p in parent)

    result = {
        "num_nodes": n,
        "num_edges": len(edges),
        "positions": pos,
        "edges": edges,
        "parent": parent,
        "type": "mesh" if is_mesh else "tree",
    }

    if graph.radii is not None:
        result["radii"] = graph.radii.detach().cpu().tolist()
    if graph.node_types is not None:
        result["node_types"] = graph.node_types.detach().cpu().tolist()

    try:
        feats = morphological_features(graph)
        result["features"] = {
            "num_branch_points": int(feats[2].item()),
            "num_tips": int(feats[3].item()),
            "strahler_order": int(feats[4].item()),
            "mean_segment_length": round(feats[5].item(), 4),
            "mean_branch_angle": round(feats[7].item(), 4),
            "total_path_length": round(feats[9].item(), 4),
            "spatial_extent": round(feats[10].item(), 4),
            "tortuosity": round(feats[11].item(), 4),
        }
    except Exception:
        result["features"] = {}

    return result


def _generate_training_data(
    data_type: str, num_graphs: int, num_nodes: int,
) -> list[SpatialGraph]:
    """Generate synthetic graphs for training."""
    graphs = []
    for _ in range(num_graphs):
        n = max(8, num_nodes + int(torch.randint(-5, 5, (1,)).item()))
        if data_type == 'neuron':
            g = random_neuron_morphology(num_nodes=n, device=DEVICE)
        else:
            g = random_branching_tree(num_nodes=n, device=DEVICE)
        graphs.append(g)
    return graphs


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

class SyntheticRequest(BaseModel):
    type: str = "tree"
    num_graphs: int = 10
    num_nodes: int = 40


@router.post("/synthetic")
def generate_synthetic(req: SyntheticRequest):
    """Generate synthetic 3D trees or neuron morphologies."""
    if req.type not in ("tree", "neuron"):
        raise HTTPException(400, "type must be 'tree' or 'neuron'")

    graphs = _generate_training_data(req.type, req.num_graphs, req.num_nodes)
    return {"graphs": [_serialize_graph(g) for g in graphs]}


# ---------------------------------------------------------------------------
# VAE training & generation
# ---------------------------------------------------------------------------

class TrainSpatialVAERequest(BaseModel):
    data_type: str = "tree"
    num_train: int = 100
    num_nodes: int = 40
    hidden_dim: int = 64
    latent_dim: int = 32
    epochs: int = 50
    lr: float = 1e-3


@router.post("/vae/train")
def train_spatial_vae_endpoint(req: TrainSpatialVAERequest):
    """Train the autoregressive spatial tree VAE."""
    graphs = _generate_training_data(req.data_type, req.num_train, req.num_nodes)
    _state['train_data'] = graphs

    max_n = max(g.num_nodes for g in graphs) + 5
    model = SpatialTreeVAE(
        latent_dim=req.latent_dim,
        hidden_dim=req.hidden_dim,
        max_nodes=max_n,
    ).to(DEVICE)

    losses = train_spatial_vae(
        model, graphs, epochs=req.epochs, lr=req.lr,
    )
    _state['vae_model'] = model

    path = CHECKPOINT_DIR / "spatial_tree_vae.pt"
    torch.save(model.state_dict(), path)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "final_loss": round(losses[-1], 4),
        "loss_curve": [round(l, 4) for l in losses],
        "num_params": n_params,
        "max_nodes": max_n,
    }


class GenerateVAERequest(BaseModel):
    num_samples: int = 4


@router.post("/vae/generate")
def generate_from_vae(req: GenerateVAERequest):
    """Generate spatial trees from the trained VAE prior."""
    if _state['vae_model'] is None:
        raise HTTPException(400, "No trained spatial VAE. Call /spatial/vae/train first.")

    model = _state['vae_model']
    generated = model.generate(num_samples=req.num_samples, device=DEVICE)
    _state['generated_vae'] = generated
    return {"graphs": [_serialize_graph(g) for g in generated]}


# ---------------------------------------------------------------------------
# Diffusion training & generation
# ---------------------------------------------------------------------------

class TrainDiffusionRequest(BaseModel):
    data_type: str = "tree"
    num_train: int = 100
    num_nodes: int = 40
    hidden_dim: int = 64
    timesteps: int = 50
    epochs: int = 50
    lr: float = 3e-4


@router.post("/diffusion/train")
def train_diffusion_endpoint(req: TrainDiffusionRequest):
    """Train the joint discrete-continuous diffusion model."""
    graphs = _generate_training_data(req.data_type, req.num_train, req.num_nodes)
    _state['train_data'] = graphs

    max_n = max(g.num_nodes for g in graphs) + 5
    model = SpatialGraphDiffusion(
        max_nodes=max_n,
        hidden_dim=req.hidden_dim,
        timesteps=req.timesteps,
    ).to(DEVICE)

    losses = train_spatial_diffusion(
        model, graphs, epochs=req.epochs, lr=req.lr,
    )
    _state['diffusion_model'] = model

    path = CHECKPOINT_DIR / "spatial_diffusion.pt"
    torch.save(model.state_dict(), path)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "final_loss": round(losses[-1], 4),
        "loss_curve": [round(l, 4) for l in losses],
        "num_params": n_params,
        "max_nodes": max_n,
    }


class GenerateDiffusionRequest(BaseModel):
    num_samples: int = 4
    num_nodes: int = 30


@router.post("/diffusion/generate")
def generate_from_diffusion(req: GenerateDiffusionRequest):
    """Generate spatial graphs via reverse diffusion."""
    if _state['diffusion_model'] is None:
        raise HTTPException(400, "No trained diffusion model. Call /spatial/diffusion/train first.")

    model = _state['diffusion_model']
    generated = []
    with torch.no_grad():
        for _ in range(req.num_samples):
            g = model.sample(
                num_nodes=min(req.num_nodes, model.max_nodes),
                device=DEVICE,
            )
            generated.append(g)

    _state['generated_diff'] = generated
    return {"graphs": [_serialize_graph(g) for g in generated]}


# ---------------------------------------------------------------------------
# Analysis / evaluation
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    model: str = "vae"
    num_samples: int = 20


@router.post("/analyze")
def analyze_model(req: AnalyzeRequest):
    """Evaluate generated samples against training data distribution."""
    if _state['train_data'] is None:
        raise HTTPException(400, "No training data. Train a model first.")

    if req.model == 'vae':
        if _state['vae_model'] is None:
            raise HTTPException(400, "No trained VAE.")
        model = _state['vae_model']
        generated = model.generate(num_samples=req.num_samples, device=DEVICE)
    elif req.model == 'diffusion':
        if _state['diffusion_model'] is None:
            raise HTTPException(400, "No trained diffusion model.")
        model = _state['diffusion_model']
        generated = []
        with torch.no_grad():
            median_n = sorted([g.num_nodes for g in _state['train_data']])[
                len(_state['train_data']) // 2
            ]
            for _ in range(req.num_samples):
                g = model.sample(
                    num_nodes=min(median_n, model.max_nodes),
                    device=DEVICE,
                )
                generated.append(g)
    else:
        raise HTTPException(400, "model must be 'vae' or 'diffusion'")

    valid = [g for g in generated if g.num_nodes >= 3 and g.num_edges >= 1]
    if not valid:
        raise HTTPException(500, "No valid samples generated.")

    metrics = full_evaluation(valid, _state['train_data'][:50])
    return {k: round(v, 6) for k, v in metrics.items()}


@router.post("/sholl/{graph_index}")
def sholl_endpoint(graph_index: int):
    """Compute Sholl profile for a generated graph."""
    generated = _state.get('generated_vae') or _state.get('generated_diff')
    if not generated:
        raise HTTPException(400, "No generated graphs. Generate samples first.")
    if graph_index < 0 or graph_index >= len(generated):
        raise HTTPException(400, f"Index out of range (have {len(generated)} graphs).")

    graph = generated[graph_index]
    radii, crossings = sholl_analysis(graph)
    return {
        "radii": radii.detach().cpu().tolist(),
        "crossings": crossings.detach().cpu().tolist(),
    }


# ---------------------------------------------------------------------------
# Interpolation (VAE only)
# ---------------------------------------------------------------------------

class InterpolateRequest(BaseModel):
    graph_idx_a: int = 0
    graph_idx_b: int = 1
    steps: int = 5


@router.post("/vae/interpolate")
def interpolate_endpoint(req: InterpolateRequest):
    """Interpolate between two training graphs in VAE latent space."""
    if _state['vae_model'] is None:
        raise HTTPException(400, "No trained VAE.")
    if _state['train_data'] is None:
        raise HTTPException(400, "No training data.")

    data = _state['train_data']
    if req.graph_idx_a >= len(data) or req.graph_idx_b >= len(data):
        raise HTTPException(400, f"Index out of range (have {len(data)} graphs).")

    model = _state['vae_model']
    g1 = data[req.graph_idx_a].to(DEVICE)
    g2 = data[req.graph_idx_b].to(DEVICE)

    interps = model.interpolate(g1, g2, steps=req.steps)
    return {
        "graphs": [_serialize_graph(g) for g in interps],
        "source": _serialize_graph(g1),
        "target": _serialize_graph(g2),
    }


# ---------------------------------------------------------------------------
# Mesh VAE: procedural meshes + latent interpolation
# ---------------------------------------------------------------------------

class MeshGenerateRequest(BaseModel):
    mesh_type: str = "showcase"
    num_meshes: int = 6


@router.post("/mesh/generate")
def generate_meshes(req: MeshGenerateRequest):
    """Generate the six canonical showcase meshes.

    Returns Cube, Octahedron, Icosahedron, Hexagonal Prism,
    3D Star, and Low-Poly Torus.
    """
    pairs = showcase_meshes(device=DEVICE)
    names = [name for name, _ in pairs]
    graphs = [g for _, g in pairs]

    _state['mesh_procedural'] = graphs
    _state['mesh_procedural_names'] = names

    serialized = []
    for name, g in zip(names, graphs):
        d = _serialize_graph(g)
        d["name"] = name
        serialized.append(d)
    return {"graphs": serialized}


class TrainMeshVAERequest(BaseModel):
    num_train: int = 100
    hidden_dim: int = 64
    latent_dim: int = 32
    epochs: int = 50
    lr: float = 1e-3


@router.post("/mesh/vae/train")
def train_mesh_vae_endpoint(req: TrainMeshVAERequest):
    """Train the mesh VAE on synthetic low-poly meshes."""
    # Auto-load showcase meshes if not already present
    if _state['mesh_procedural'] is None:
        pairs = showcase_meshes(device=DEVICE)
        _state['mesh_procedural'] = [g for _, g in pairs]
        _state['mesh_procedural_names'] = [name for name, _ in pairs]

    graphs = generate_mesh_dataset(num=req.num_train, device=DEVICE)

    # Prepend the showcase meshes (repeated) so the encoder learns
    # their latent representations well — these are used for interpolation.
    showcase = list(_state['mesh_procedural'])
    showcase_repeated = showcase * max(1, req.num_train // (len(showcase) * 3))
    graphs = showcase_repeated + graphs
    _state['mesh_train_data'] = graphs

    max_n = max(g.num_nodes for g in graphs) + 4
    model = SpatialMeshVAE(
        latent_dim=req.latent_dim,
        hidden_dim=req.hidden_dim,
        max_nodes=max_n,
    ).to(DEVICE)

    losses = train_mesh_vae(
        model, graphs, epochs=req.epochs, lr=req.lr,
    )
    _state['mesh_vae_model'] = model

    path = CHECKPOINT_DIR / "mesh_vae.pt"
    torch.save(model.state_dict(), path)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "final_loss": round(losses[-1], 4),
        "loss_curve": [round(l, 4) for l in losses],
        "num_params": n_params,
        "max_nodes": max_n,
    }


class GenerateMeshVAERequest(BaseModel):
    num_samples: int = 4


@router.post("/mesh/vae/generate")
def generate_from_mesh_vae(req: GenerateMeshVAERequest):
    """Generate meshes from the trained mesh VAE prior."""
    if _state['mesh_vae_model'] is None:
        raise HTTPException(400, "No trained mesh VAE. Call /spatial/mesh/vae/train first.")

    model = _state['mesh_vae_model']
    generated = model.generate(num_samples=req.num_samples, device=DEVICE)
    return {"graphs": [_serialize_graph(g) for g in generated]}


class MeshInterpolateRequest(BaseModel):
    graph_idx_a: int = 0
    graph_idx_b: int = 1
    steps: int = 7


@router.post("/mesh/vae/interpolate")
def mesh_interpolate_endpoint(req: MeshInterpolateRequest):
    """Interpolate between two procedural meshes.

    Uses direct geometric blending (position lerp + edge topology
    transition with greedy node matching) for deterministic, smooth
    morphs that work without a trained VAE.
    """
    if _state['mesh_procedural'] is None:
        # Auto-load showcase shapes
        pairs = showcase_meshes(device=DEVICE)
        _state['mesh_procedural'] = [g for _, g in pairs]
        _state['mesh_procedural_names'] = [name for name, _ in pairs]

    data = _state['mesh_procedural']
    if req.graph_idx_a >= len(data) or req.graph_idx_b >= len(data):
        raise HTTPException(400, f"Index out of range (have {len(data)} procedural meshes).")

    names = _state.get('mesh_procedural_names') or []
    name_a = names[req.graph_idx_a] if req.graph_idx_a < len(names) else f"#{req.graph_idx_a}"
    name_b = names[req.graph_idx_b] if req.graph_idx_b < len(names) else f"#{req.graph_idx_b}"

    g1 = data[req.graph_idx_a]
    g2 = data[req.graph_idx_b]

    interps = geometric_interpolate(g1, g2, steps=req.steps)

    # Replace endpoints with the exact original meshes (original
    # node count, not padded) for pixel-perfect source/target display.
    interps[0] = g1
    interps[-1] = g2

    return {
        "graphs": [_serialize_graph(g) for g in interps],
        "source": _serialize_graph(g1),
        "target": _serialize_graph(g2),
        "source_name": name_a,
        "target_name": name_b,
    }
