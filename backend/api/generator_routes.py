from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch

from backend.core.generator.model import (
    ConditionalGraphGenerator, build_training_set, train_generator, sample_graphs,
)
from backend.core.graph_utils import graph_density, clustering_coefficient, adj_to_edge_index
from backend.config import DEVICE, CHECKPOINT_DIR, GENERATOR_DEFAULTS

router = APIRouter(prefix="/generator", tags=["generator"])

_state = {"model": None}


class TrainRequest(BaseModel):
    max_nodes: int = GENERATOR_DEFAULTS["max_nodes"]
    latent_dim: int = GENERATOR_DEFAULTS["latent_dim"]
    hidden_dim: int = GENERATOR_DEFAULTS["hidden_dim"]
    lr: float = GENERATOR_DEFAULTS["lr"]
    epochs: int = GENERATOR_DEFAULTS["epochs"]
    num_training_graphs: int = GENERATOR_DEFAULTS["num_training_graphs"]


class GenerateRequest(BaseModel):
    num_nodes: float = 0.5
    density: float = 0.3
    clustering: float = 0.3
    num_samples: int = 4
    threshold: float = 0.5


@router.post("/train")
def train(req: TrainRequest):
    """Build a dataset of random graphs and train the conditional generator."""
    dataset = build_training_set(req.num_training_graphs, req.max_nodes)

    model = ConditionalGraphGenerator(
        max_nodes=req.max_nodes,
        latent_dim=req.latent_dim,
        hidden_dim=req.hidden_dim,
    ).to(DEVICE)

    losses = train_generator(model, dataset, epochs=req.epochs, lr=req.lr)
    _state["model"] = model

    path = CHECKPOINT_DIR / "graph_generator.pt"
    torch.save(model.state_dict(), path)

    return {
        "final_loss": round(losses[-1], 4),
        "loss_curve": [round(l, 4) for l in losses[::10]],
        "num_training_graphs": req.num_training_graphs,
        "checkpoint": str(path),
    }


@router.post("/generate")
def generate(req: GenerateRequest):
    """Generate graphs conditioned on target properties."""
    if _state["model"] is None:
        raise HTTPException(400, "No trained model. Call /generator/train first.")

    model = _state["model"]
    conditions = torch.tensor(
        [[req.num_nodes, req.density, req.clustering]],
        dtype=torch.float32,
    )

    graphs = sample_graphs(model, conditions, req.num_samples, threshold=req.threshold)

    results = []
    for adj in graphs:
        # trim padding: find actual number of active nodes
        deg = adj.sum(dim=1)
        active = (deg > 0).sum().item()
        if active == 0:
            active = adj.size(0)

        adj_trimmed = adj[:active, :active]
        edge_index = adj_to_edge_index(adj_trimmed)

        results.append({
            "num_nodes": active,
            "num_edges": int(edge_index.size(1) // 2),
            "edges": edge_index.t().tolist(),
            "density": round(graph_density(adj_trimmed), 4),
            "avg_clustering": round(float(clustering_coefficient(adj_trimmed).mean().item()), 4),
        })

    return {"graphs": results, "conditions": {
        "num_nodes": req.num_nodes,
        "density": req.density,
        "clustering": req.clustering,
    }}
