from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch

from backend.core.gnn.model import (
    NodeClassifier, GraphClassifier, train_node_classifier, generate_karate_club,
)
from backend.core.gnn.layers import GCNLayer, GATLayer
from backend.core.graph_utils import adj_to_edge_index
from backend.config import DEVICE, CHECKPOINT_DIR, GNN_DEFAULTS

router = APIRouter(prefix="/gnn", tags=["gnn"])

# in-memory state for the current session
_state = {"model": None, "adj": None, "features": None, "labels": None}


class TrainRequest(BaseModel):
    layer_type: str = "gcn"
    hidden_dim: int = GNN_DEFAULTS["hidden_dim"]
    num_layers: int = GNN_DEFAULTS["num_layers"]
    dropout: float = GNN_DEFAULTS["dropout"]
    lr: float = GNN_DEFAULTS["lr"]
    epochs: int = GNN_DEFAULTS["epochs"]
    train_nodes: list[int] = [0, 1, 2, 33]


class PredictRequest(BaseModel):
    node_ids: list[int] | None = None


@router.get("/karate-club")
def get_karate_club():
    """Load the Karate Club graph and return its structure."""
    adj, features, labels = generate_karate_club()
    _state["adj"] = adj
    _state["features"] = features
    _state["labels"] = labels

    edge_index = adj_to_edge_index(adj)
    return {
        "num_nodes": adj.size(0),
        "num_edges": int(edge_index.size(1) // 2),
        "edges": edge_index.t().tolist(),
        "labels": labels.tolist(),
        "feature_dim": features.size(1),
    }


@router.post("/train")
def train(req: TrainRequest):
    """Train a node classifier on the Karate Club graph."""
    if _state["adj"] is None:
        # auto-load if not yet loaded
        get_karate_club()

    adj = _state["adj"].to(DEVICE)
    features = _state["features"].to(DEVICE)
    labels = _state["labels"].to(DEVICE)

    train_mask = torch.zeros(adj.size(0), dtype=torch.bool, device=DEVICE)
    for idx in req.train_nodes:
        if 0 <= idx < adj.size(0):
            train_mask[idx] = True

    if train_mask.sum() == 0:
        raise HTTPException(400, "Need at least one training node.")

    model = NodeClassifier(
        in_features=features.size(1),
        hidden=req.hidden_dim,
        num_classes=int(labels.max().item()) + 1,
        n_layers=req.num_layers,
        dropout=req.dropout,
        layer_type=req.layer_type,
    ).to(DEVICE)

    losses, accs = train_node_classifier(
        model, adj, features, labels, train_mask,
        lr=req.lr, epochs=req.epochs,
    )
    _state["model"] = model

    # save checkpoint
    path = CHECKPOINT_DIR / "gnn_node_classifier.pt"
    torch.save(model.state_dict(), path)

    return {
        "final_loss": round(losses[-1], 4),
        "final_accuracy": round(accs[-1], 4),
        "loss_curve": [round(l, 4) for l in losses[::5]],
        "accuracy_curve": [round(a, 4) for a in accs[::5]],
        "checkpoint": str(path),
    }


@router.post("/predict")
def predict(req: PredictRequest):
    """Run inference on the trained model."""
    if _state["model"] is None:
        raise HTTPException(400, "No trained model. Call /gnn/train first.")

    model = _state["model"]
    adj = _state["adj"].to(DEVICE)
    features = _state["features"].to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

    node_ids = req.node_ids if req.node_ids else list(range(adj.size(0)))
    results = []
    for i in node_ids:
        if 0 <= i < adj.size(0):
            results.append({
                "node": i,
                "predicted": int(preds[i].item()),
                "true_label": int(_state["labels"][i].item()),
                "confidence": round(float(probs[i].max().item()), 4),
            })

    correct = sum(1 for r in results if r["predicted"] == r["true_label"])
    return {
        "predictions": results,
        "accuracy": round(correct / len(results), 4) if results else 0.0,
    }
