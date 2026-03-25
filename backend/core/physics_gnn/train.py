"""
Training utilities and experiment runner for physics-informed GNNs.

Provides:
    - Training loops for both classification/regression and generation
    - Ablation study runner to isolate the contribution of each component
    - Experiment logging with structured metrics
    - Integration with the existing SpatialGraph data format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import math
from typing import Optional
from dataclasses import dataclass, field

from .models import PhysicsInformedGNN, PhysicsInformedGraphGenerator


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    physics_weight: float = 0.01
    warmup_epochs: int = 10
    patience: int = 30
    min_lr: float = 1e-6
    log_interval: int = 10


# ---------------------------------------------------------------------------
# Node-level training (classification / regression)
# ---------------------------------------------------------------------------

def train_node_model(
    model: PhysicsInformedGNN,
    positions: Tensor,
    adj: Tensor,
    labels: Tensor,
    train_mask: Tensor,
    val_mask: Optional[Tensor] = None,
    x: Optional[Tensor] = None,
    faces: Optional[Tensor] = None,
    config: Optional[TrainConfig] = None,
    task_loss: str = 'cross_entropy',
) -> dict:
    """Train a PhysicsInformedGNN for node-level prediction.

    Args:
        model: the network.
        positions: (N, 3) node coordinates.
        adj: (N, N) adjacency.
        labels: (N,) targets (class indices or regression values).
        train_mask: (N,) boolean training mask.
        val_mask: (N,) boolean validation mask.
        x: (N, F) optional node features.
        faces: (F_tri, 3) optional triangle faces.
        config: training hyperparameters.
        task_loss: 'cross_entropy' or 'mse'.

    Returns:
        dict with 'train_losses', 'val_losses', 'best_epoch', 'best_val_loss'.
    """
    cfg = config or TrainConfig()
    device = positions.device
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)

    loss_fn = F.cross_entropy if task_loss == 'cross_entropy' else F.mse_loss

    history = {'train_losses': [], 'val_losses': [], 'best_epoch': 0, 'best_val_loss': float('inf')}
    best_state = None
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()

        # Physics weight warmup
        physics_w = cfg.physics_weight * min(1.0, epoch / max(cfg.warmup_epochs, 1))

        output, energy = model(
            positions, adj, x=x, faces=faces, return_energy=True,
        )

        # Task loss on training nodes
        if task_loss == 'cross_entropy':
            task = loss_fn(output[train_mask], labels[train_mask])
        else:
            task = loss_fn(output[train_mask].squeeze(), labels[train_mask].float())

        loss = task + physics_w * energy

        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        history['train_losses'].append(loss.item())

        # Validation
        if val_mask is not None and val_mask.sum() > 0:
            model.eval()
            with torch.no_grad():
                val_out = model(positions, adj, x=x, faces=faces)
                if task_loss == 'cross_entropy':
                    val_loss = loss_fn(val_out[val_mask], labels[val_mask]).item()
                else:
                    val_loss = loss_fn(val_out[val_mask].squeeze(), labels[val_mask].float()).item()

            history['val_losses'].append(val_loss)

            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                break

        if epoch % cfg.log_interval == 0:
            val_str = f", val={history['val_losses'][-1]:.4f}" if history['val_losses'] else ""
            print(f"  epoch {epoch:4d}  loss={loss.item():.4f}  task={task.item():.4f}  "
                  f"energy={energy.item():.4f}{val_str}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ---------------------------------------------------------------------------
# Generator training
# ---------------------------------------------------------------------------

def train_generator(
    model: PhysicsInformedGraphGenerator,
    graphs: list,
    config: Optional[TrainConfig] = None,
) -> dict:
    """Train the physics-informed graph generator.

    Each graph should have .pos (N,3), .adj (N,N), and optionally a mask.

    Args:
        model: the generator network.
        graphs: list of SpatialGraph objects.
        config: training hyperparameters.

    Returns:
        dict with 'losses', 'best_epoch', 'best_loss'.
    """
    cfg = config or TrainConfig(epochs=300, lr=3e-4)
    device = next(model.parameters()).device

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)

    history = {'losses': [], 'best_epoch': 0, 'best_loss': float('inf')}
    best_state = None

    N_max = model.max_nodes

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0

        for graph in graphs:
            optimizer.zero_grad()

            # Pad graph to max_nodes
            n = graph.pos.size(0)
            if n > N_max:
                continue

            pos_padded = torch.zeros(N_max, 3, device=device)
            pos_padded[:n] = graph.pos.to(device)

            adj_padded = torch.zeros(N_max, N_max, device=device)
            adj_padded[:n, :n] = graph.adj.to(device)

            mask = torch.zeros(N_max, device=device)
            mask[:n] = 1.0

            # Sample latent code (encoder-free training: random z + reconstruction)
            z = torch.randn(model.latent_dim, device=device)

            output = model(z, pos_padded, adj_padded, mask)
            loss = output['loss']

            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(graphs), 1)
        history['losses'].append(avg_loss)

        if avg_loss < history['best_loss']:
            history['best_loss'] = avg_loss
            history['best_epoch'] = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % cfg.log_interval == 0:
            print(f"  epoch {epoch:4d}  loss={avg_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Result of one ablation experiment."""
    name: str
    config: dict
    train_loss: float
    val_loss: float
    val_metric: float  # accuracy or MSE depending on task
    train_time: float
    num_params: int


def run_ablation(
    positions: Tensor,
    adj: Tensor,
    labels: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    x: Optional[Tensor] = None,
    faces: Optional[Tensor] = None,
    task_loss: str = 'cross_entropy',
    epochs: int = 100,
) -> list[AblationResult]:
    """Run ablation study comparing different component combinations.

    Tests the contribution of each physics-informed component by
    systematically disabling them:
        1. Full model (all components)
        2. No curvature features
        3. No diffusion filters
        4. No reaction-diffusion layers
        5. No curvature attention
        6. No physics regularisation
        7. Vanilla GCN baseline (no physics components)

    Returns a list of AblationResult for comparison.
    """
    device = positions.device
    n_classes = labels.max().item() + 1 if task_loss == 'cross_entropy' else 1

    configs = [
        ('Full PI-GNN', dict(
            use_curvature=True, use_diffusion=True,
            use_reaction_diffusion=True, use_attention=True, regularise=True,
        )),
        ('No curvature', dict(
            use_curvature=False, use_diffusion=True,
            use_reaction_diffusion=True, use_attention=True, regularise=True,
        )),
        ('No diffusion', dict(
            use_curvature=True, use_diffusion=False,
            use_reaction_diffusion=True, use_attention=True, regularise=True,
        )),
        ('No reaction-diffusion', dict(
            use_curvature=True, use_diffusion=True,
            use_reaction_diffusion=False, use_attention=True, regularise=True,
        )),
        ('No attention', dict(
            use_curvature=True, use_diffusion=True,
            use_reaction_diffusion=True, use_attention=False, regularise=True,
        )),
        ('No regularisation', dict(
            use_curvature=True, use_diffusion=True,
            use_reaction_diffusion=True, use_attention=True, regularise=False,
        )),
        ('Vanilla (no physics)', dict(
            use_curvature=False, use_diffusion=False,
            use_reaction_diffusion=False, use_attention=False, regularise=False,
        )),
    ]

    in_channels = x.size(1) if x is not None else 0
    results = []

    for name, model_cfg in configs:
        print(f"\n  Ablation: {name}")
        model = PhysicsInformedGNN(
            in_channels=in_channels,
            hidden_channels=32,
            out_channels=n_classes,
            num_layers=3,
            task='node',
            **model_cfg,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        t0 = time.time()
        history = train_node_model(
            model, positions, adj, labels, train_mask, val_mask,
            x=x, faces=faces,
            config=TrainConfig(epochs=epochs, log_interval=epochs + 1),
            task_loss=task_loss,
        )
        train_time = time.time() - t0

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(positions, adj, x=x, faces=faces)

            if task_loss == 'cross_entropy':
                pred = out.argmax(dim=1)
                metric = (pred[val_mask] == labels[val_mask]).float().mean().item()
            else:
                metric = F.mse_loss(out[val_mask].squeeze(), labels[val_mask].float()).item()

        results.append(AblationResult(
            name=name,
            config=model_cfg,
            train_loss=history['train_losses'][-1],
            val_loss=history['val_losses'][-1] if history['val_losses'] else float('nan'),
            val_metric=metric,
            train_time=train_time,
            num_params=n_params,
        ))

        metric_name = "accuracy" if task_loss == 'cross_entropy' else "MSE"
        print(f"    {metric_name}={metric:.4f}  params={n_params}  time={train_time:.1f}s")

    return results


def print_ablation_table(results: list[AblationResult], metric_name: str = 'accuracy'):
    """Pretty-print ablation results as a table."""
    print(f"\n{'Model':<28} {'Params':>8} {metric_name:>10} {'Train':>8} {'Val':>8} {'Time':>6}")
    print('-' * 78)
    for r in results:
        print(f"{r.name:<28} {r.num_params:>8} {r.val_metric:>10.4f} "
              f"{r.train_loss:>8.4f} {r.val_loss:>8.4f} {r.train_time:>5.1f}s")
