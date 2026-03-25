"""Training utilities for hyperbolic GNNs and embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
import time
from typing import Optional
from dataclasses import dataclass

from .manifolds import PoincareBall, ManifoldParameter, RiemannianAdam, get_device
from .models import HyperbolicGNN, HyperbolicEmbedding


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class HyperbolicTrainConfig:
    """Training hyperparameters for hyperbolic models."""
    epochs: int = 200
    lr: float = 1e-2
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    patience: int = 30
    log_interval: int = 10


# ---------------------------------------------------------------------------
# Hyperbolic GNN Training
# ---------------------------------------------------------------------------

def train_hyperbolic_gnn(
    model: HyperbolicGNN,
    features: Tensor,
    adj: Tensor,
    labels: Tensor,
    config: Optional[HyperbolicTrainConfig] = None,
    train_mask: Optional[Tensor] = None,
    val_mask: Optional[Tensor] = None,
) -> dict:
    """Train a HyperbolicGNN for node classification.

    Args:
        model: HyperbolicGNN instance.
        features: (N, F) node features.
        adj: (N, N) dense adjacency.
        labels: (N,) class labels.
        config: training config.
        train_mask: (N,) boolean mask for training nodes.
        val_mask: (N,) boolean mask for validation nodes.
    Returns:
        dict with 'losses', 'accuracies', 'embeddings'.
    """
    cfg = config or HyperbolicTrainConfig()
    device = get_device()

    model = model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)

    if train_mask is None:
        train_mask = torch.ones(features.size(0), dtype=torch.bool, device=device)
    else:
        train_mask = train_mask.to(device)

    if val_mask is not None:
        val_mask = val_mask.to(device)

    # Separate Riemannian and Euclidean params
    manifold_params = []
    euclidean_params = []
    for p in model.parameters():
        if hasattr(p, 'manifold') and p.manifold is not None:
            manifold_params.append(p)
        else:
            euclidean_params.append(p)

    optimizers = []
    if manifold_params:
        optimizers.append(RiemannianAdam(manifold_params, lr=cfg.lr))
    if euclidean_params:
        optimizers.append(AdamW(euclidean_params, lr=cfg.lr, weight_decay=cfg.weight_decay))

    history = {'losses': [], 'accuracies': [], 'embeddings': None}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        for opt in optimizers:
            opt.zero_grad()

        logits = model(features, adj)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()

        # Train accuracy
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean().item()

        history['losses'].append(loss.item())
        history['accuracies'].append(train_acc)

        # Validation
        if val_mask is not None and val_mask.sum() > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(features, adj)
                val_pred = val_logits.argmax(dim=1)
                val_acc = (val_pred[val_mask] == labels[val_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                break

        if epoch % cfg.log_interval == 0:
            val_str = f"  val_acc={val_acc:.4f}" if val_mask is not None else ""
            print(f"  epoch {epoch:4d}  loss={loss.item():.4f}  "
                  f"train_acc={train_acc:.4f}{val_str}")

    # Final embeddings
    model.eval()
    with torch.no_grad():
        history['embeddings'] = model.get_embeddings(features, adj).cpu()

    return history


# ---------------------------------------------------------------------------
# Hyperbolic Embedding Training
# ---------------------------------------------------------------------------

def train_hyperbolic_embedding(
    num_nodes: int,
    edges: Tensor,
    embed_dim: int = 16,
    c: float = 1.0,
    epochs: int = 300,
    lr: float = 1e-2,
    neg_ratio: int = 5,
) -> dict:
    """Train Poincare embeddings for link prediction.

    Args:
        num_nodes: number of nodes.
        edges: (E, 2) edge index tensor.
        embed_dim: embedding dimension.
        c: curvature parameter.
        epochs: training epochs.
        lr: learning rate.
        neg_ratio: negative samples per positive edge.
    Returns:
        dict with 'losses', 'embeddings', 'model'.
    """
    device = get_device()
    edges = edges.to(device)

    model = HyperbolicEmbedding(
        num_nodes=num_nodes,
        embed_dim=embed_dim,
        c=c,
        init_scale=0.01,
    ).to(device)

    optimizer = RiemannianAdam(model.parameters(), lr=lr)

    history = {'losses': [], 'embeddings': None, 'model': model}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Sample negative edges
        neg_src = torch.randint(0, num_nodes, (edges.size(0) * neg_ratio,), device=device)
        neg_dst = torch.randint(0, num_nodes, (edges.size(0) * neg_ratio,), device=device)
        neg_edges = torch.stack([neg_src, neg_dst], dim=1)

        loss = model.loss(edges, neg_edges)
        loss.backward()
        optimizer.step()

        history['losses'].append(loss.item())

        if epoch % 50 == 0:
            print(f"  epoch {epoch:4d}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        history['embeddings'] = model.get_embeddings().cpu()

    return history


# ---------------------------------------------------------------------------
# Euclidean vs Hyperbolic Embedding Comparison
# ---------------------------------------------------------------------------

def compare_embeddings(
    num_nodes: int,
    edges: Tensor,
    embed_dim: int = 16,
    epochs: int = 300,
    lr: float = 1e-2,
) -> dict:
    """Train both Euclidean and Poincare embeddings, compare distortion.

    Args:
        num_nodes: number of nodes.
        edges: (E, 2) edge index tensor.
        embed_dim: embedding dimension.
        epochs: training epochs.
        lr: learning rate.
    Returns:
        dict with 'euclidean' and 'poincare' sub-dicts containing
        embeddings and distortion metrics.
    """
    device = get_device()
    edges = edges.to(device)

    # --- Euclidean embeddings ---
    euclidean_emb = nn.Parameter(
        0.01 * torch.randn(num_nodes, embed_dim, dtype=torch.float32, device=device)
    )
    euc_optimizer = AdamW([euclidean_emb], lr=lr)

    euc_losses = []
    for epoch in range(epochs):
        euc_optimizer.zero_grad()

        src_e = euclidean_emb[edges[:, 0]]
        dst_e = euclidean_emb[edges[:, 1]]
        pos_dist = torch.norm(src_e - dst_e, dim=-1)

        neg_src = torch.randint(0, num_nodes, (edges.size(0),), device=device)
        neg_dst = torch.randint(0, num_nodes, (edges.size(0),), device=device)
        neg_dist = torch.norm(
            euclidean_emb[neg_src] - euclidean_emb[neg_dst], dim=-1
        )

        # Margin-based loss
        loss_e = F.relu(pos_dist - neg_dist + 1.0).mean()
        loss_e.backward()
        euc_optimizer.step()
        euc_losses.append(loss_e.item())

    # --- Poincare embeddings ---
    hyp_result = train_hyperbolic_embedding(
        num_nodes, edges, embed_dim=embed_dim, c=1.0,
        epochs=epochs, lr=lr,
    )

    # --- Compute distortion metrics ---
    with torch.no_grad():
        # Graph shortest-path approximation via hop distance for connected edges
        # Use edge distances as proxy
        euc_final = euclidean_emb.detach()
        hyp_final = hyp_result['embeddings'].to(device)

        manifold = PoincareBall(c=1.0)

        # Distortion on edges
        src_idx = edges[:, 0]
        dst_idx = edges[:, 1]

        euc_edge_dist = torch.norm(
            euc_final[src_idx] - euc_final[dst_idx], dim=-1
        )
        hyp_edge_dist = manifold.dist(
            hyp_final[src_idx], hyp_final[dst_idx]
        ).squeeze(-1)

        # Mean average precision proxy: ratio of edge distances
        euc_mean_dist = euc_edge_dist.mean().item()
        hyp_mean_dist = hyp_edge_dist.mean().item()

        # Distortion: std of pairwise distance ratios
        euc_ratio = euc_edge_dist / euc_edge_dist.mean().clamp(min=1e-8)
        hyp_ratio = hyp_edge_dist / hyp_edge_dist.mean().clamp(min=1e-8)

        euc_distortion = euc_ratio.std().item()
        hyp_distortion = hyp_ratio.std().item()

    return {
        'euclidean': {
            'embeddings': euc_final.cpu(),
            'losses': euc_losses,
            'mean_edge_dist': euc_mean_dist,
            'distortion': euc_distortion,
        },
        'poincare': {
            'embeddings': hyp_final.cpu(),
            'losses': hyp_result['losses'],
            'mean_edge_dist': hyp_mean_dist,
            'distortion': hyp_distortion,
        },
    }
