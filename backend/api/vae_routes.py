from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch

from backend.core.graphvae.model import (
    GraphVAE, DenoisingDiffusion, train_vae, train_diffusion,
    generate_social_skeletons, interpolate_latent,
)
from backend.core.graph_utils import (
    graph_density, clustering_coefficient, adj_to_edge_index, one_hot_degree_features,
)
from backend.config import DEVICE, CHECKPOINT_DIR, VAE_DEFAULTS, DIFFUSION_DEFAULTS

router = APIRouter(prefix="/vae", tags=["vae"])

_state = {"vae": None, "diffusion": None, "train_graphs": None}


class TrainVAERequest(BaseModel):
    max_nodes: int = VAE_DEFAULTS["max_nodes"]
    latent_dim: int = VAE_DEFAULTS["latent_dim"]
    hidden_dim: int = VAE_DEFAULTS["hidden_dim"]
    lr: float = VAE_DEFAULTS["lr"]
    epochs: int = VAE_DEFAULTS["epochs"]
    num_graphs: int = 200
    graph_sizes: list[int] = [8, 10, 12, 15]


class TrainDiffusionRequest(BaseModel):
    max_nodes: int = DIFFUSION_DEFAULTS["max_nodes"]
    hidden_dim: int = DIFFUSION_DEFAULTS["hidden_dim"]
    timesteps: int = DIFFUSION_DEFAULTS["timesteps"]
    lr: float = DIFFUSION_DEFAULTS["lr"]
    epochs: int = DIFFUSION_DEFAULTS["epochs"]
    num_graphs: int = 200


class InterpolateRequest(BaseModel):
    graph_idx_a: int = 0
    graph_idx_b: int = 1
    steps: int = 5


def _graph_to_dict(adj):
    deg = adj.sum(dim=1)
    active = max(int((deg > 0).sum().item()), 2)
    adj_t = adj[:active, :active]
    ei = adj_to_edge_index(adj_t)
    return {
        "num_nodes": active,
        "num_edges": int(ei.size(1) // 2),
        "edges": ei.t().tolist(),
        "density": round(graph_density(adj_t), 4),
        "avg_clustering": round(float(clustering_coefficient(adj_t).mean().item()), 4),
    }


@router.post("/train")
def train_vae_endpoint(req: TrainVAERequest):
    """Train the Graph VAE on synthetic social network skeletons."""
    graphs = generate_social_skeletons(req.num_graphs, req.graph_sizes)
    _state["train_graphs"] = graphs

    # feature dim comes from one_hot_degree_features(max_degree=10) → 11
    node_feat_dim = graphs[0][0].size(1)

    model = GraphVAE(
        max_nodes=req.max_nodes,
        node_feat_dim=node_feat_dim,
        latent_dim=req.latent_dim,
        hidden_dim=req.hidden_dim,
    ).to(DEVICE)

    losses = train_vae(model, graphs, epochs=req.epochs, lr=req.lr)
    _state["vae"] = model

    path = CHECKPOINT_DIR / "graph_vae.pt"
    torch.save(model.state_dict(), path)

    return {
        "final_loss": round(losses[-1], 4),
        "loss_curve": [round(l, 4) for l in losses[::10]],
        "num_training_graphs": len(graphs),
        "checkpoint": str(path),
    }


@router.post("/generate")
def generate_vae(num_samples: int = 4):
    """Sample new graphs from the VAE prior."""
    if _state["vae"] is None:
        raise HTTPException(400, "No trained VAE. Call /vae/train first.")

    model = _state["vae"]
    model.eval()

    results = []
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(model.latent_dim, device=DEVICE)
            adj_recon, _ = model.decoder(z)
            adj_bin = (adj_recon.squeeze(0) > 0.5).float()
            # ensure symmetric
            adj_bin = torch.triu(adj_bin, diagonal=1)
            adj_bin = adj_bin + adj_bin.t()
            results.append(_graph_to_dict(adj_bin))

    return {"graphs": results}


@router.post("/interpolate")
def interpolate(req: InterpolateRequest):
    """Interpolate between two training graphs in latent space."""
    if _state["vae"] is None:
        raise HTTPException(400, "No trained VAE. Call /vae/train first.")
    if _state["train_graphs"] is None:
        raise HTTPException(400, "No training graphs available.")

    graphs = _state["train_graphs"]
    if req.graph_idx_a >= len(graphs) or req.graph_idx_b >= len(graphs):
        raise HTTPException(400, f"Graph index out of range (have {len(graphs)} graphs).")

    model = _state["vae"]
    feat_a, adj_a = graphs[req.graph_idx_a]
    feat_b, adj_b = graphs[req.graph_idx_b]

    interps = interpolate_latent(model, (feat_a, adj_a), (feat_b, adj_b), steps=req.steps)
    return {
        "steps": [_graph_to_dict(adj) for adj, _ in interps],
        "source": _graph_to_dict(adj_a),
        "target": _graph_to_dict(adj_b),
    }


@router.post("/diffusion/train")
def train_diffusion_endpoint(req: TrainDiffusionRequest):
    """Train the graph diffusion model."""
    graphs = generate_social_skeletons(req.num_graphs, [8, 10, 12])

    model = DenoisingDiffusion(
        max_nodes=req.max_nodes,
        hidden_dim=req.hidden_dim,
        timesteps=req.timesteps,
    ).to(DEVICE)

    losses = train_diffusion(model, graphs, epochs=req.epochs, lr=req.lr)
    _state["diffusion"] = model

    path = CHECKPOINT_DIR / "graph_diffusion.pt"
    torch.save(model.state_dict(), path)

    return {
        "final_loss": round(losses[-1], 4),
        "loss_curve": [round(l, 4) for l in losses[::10]],
        "checkpoint": str(path),
    }


@router.post("/diffusion/generate")
def generate_diffusion(num_samples: int = 4):
    """Sample graphs via iterative denoising."""
    if _state["diffusion"] is None:
        raise HTTPException(400, "No trained diffusion model. Call /vae/diffusion/train first.")

    model = _state["diffusion"]
    model.eval()

    results = []
    with torch.no_grad():
        for _ in range(num_samples):
            adj = model.sample(device=DEVICE)
            results.append(_graph_to_dict(adj))

    return {"graphs": results}
