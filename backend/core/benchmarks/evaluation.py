"""
Benchmarking framework for spatial graph generative models.

Provides a unified pipeline for:
    1. Generating synthetic training data (multiple regimes)
    2. Training models with consistent configurations
    3. Evaluating against reference sets using the full metric suite
    4. Comparing models head-to-head

This is the proper way to evaluate generative models on graphs --
not FID on adjacency matrices (which ignores spatial structure),
not Chamfer distance on positions (which ignores topology).

Usage:
    results = run_benchmark(
        models={'vae': vae_model, 'diffusion': diff_model},
        data_config=DataConfig(regime='neuron', num_train=200, num_test=50),
        train_config=TrainConfig(epochs=100),
    )
    print(compare_results(results))
"""

import torch
import torch.nn as nn
import time
from dataclasses import dataclass, field
from typing import Literal

from ..spatial.graph3d import SpatialGraph
from ..spatial.synthetic import random_branching_tree, random_neuron_morphology
from ..spatial.metrics import full_evaluation, morphological_features
from ..spatial.tree_gen import SpatialTreeVAE, train_spatial_vae
from ..spatial.diffusion3d import SpatialGraphDiffusion, train_spatial_diffusion


@dataclass
class DataConfig:
    """Configuration for synthetic dataset generation."""
    regime: Literal['tree', 'neuron', 'mixed'] = 'tree'
    num_train: int = 200
    num_test: int = 50
    num_nodes: int = 40
    seed: int = 42
    device: torch.device | None = None


@dataclass
class TrainConfig:
    """Configuration for model training."""
    epochs: int = 100
    lr: float = 1e-3
    eval_every: int = 25
    patience: int = 20


@dataclass
class BenchmarkResult:
    """Result from evaluating a single model."""
    model_name: str
    metrics: dict[str, float]
    train_losses: list[float]
    train_time: float
    num_params: int


def generate_dataset(
    config: DataConfig,
) -> tuple[list[SpatialGraph], list[SpatialGraph]]:
    """Create train/test splits of synthetic spatial graphs.

    Returns:
        (train_graphs, test_graphs)
    """
    torch.manual_seed(config.seed)
    total = config.num_train + config.num_test
    graphs: list[SpatialGraph] = []

    if config.regime == 'tree':
        for _ in range(total):
            n = max(8, config.num_nodes + int(torch.randint(-10, 10, (1,)).item()))
            g = random_branching_tree(
                num_nodes=n,
                branch_prob=0.12 + 0.1 * torch.rand(1).item(),
                device=config.device,
            )
            graphs.append(g)

    elif config.regime == 'neuron':
        for _ in range(total):
            n = max(15, config.num_nodes + int(torch.randint(-15, 15, (1,)).item()))
            g = random_neuron_morphology(
                num_nodes=n,
                num_dendrites=3 + int(torch.randint(0, 3, (1,)).item()),
                device=config.device,
            )
            graphs.append(g)

    elif config.regime == 'mixed':
        half = total // 2
        for _ in range(half):
            n = max(8, config.num_nodes + int(torch.randint(-10, 10, (1,)).item()))
            graphs.append(random_branching_tree(num_nodes=n, device=config.device))
        for _ in range(total - half):
            n = max(15, config.num_nodes + int(torch.randint(-10, 10, (1,)).item()))
            graphs.append(random_neuron_morphology(num_nodes=n, device=config.device))
        # shuffle
        perm = torch.randperm(len(graphs)).tolist()
        graphs = [graphs[i] for i in perm]

    train = graphs[:config.num_train]
    test = graphs[config.num_train:]
    return train, test


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(
    model: nn.Module,
    test_graphs: list[SpatialGraph],
    num_samples: int = 50,
) -> dict[str, float]:
    """Generate samples from a trained model and run full evaluation.

    Handles both SpatialTreeVAE (has .generate()) and
    SpatialGraphDiffusion (has .sample()).
    """
    model.eval()
    device = next(model.parameters()).device
    generated: list[SpatialGraph] = []

    with torch.no_grad():
        if hasattr(model, 'generate'):
            generated = model.generate(num_samples=num_samples, device=device)
        elif hasattr(model, 'sample'):
            for _ in range(num_samples):
                # sample with median node count from test set
                median_n = sorted([g.num_nodes for g in test_graphs])[len(test_graphs) // 2]
                g = model.sample(num_nodes=min(median_n, model.max_nodes), device=device)
                generated.append(g)
        else:
            raise ValueError(f"Model {type(model).__name__} has no generate() or sample() method.")

    # filter out degenerate samples (< 3 nodes or empty adjacency)
    valid = [g for g in generated if g.num_nodes >= 3 and g.num_edges >= 1]
    if not valid:
        return {'mmd': float('inf'), 'note': 'no valid samples generated'}

    return full_evaluation(valid, test_graphs)


def _train_model(
    model: nn.Module,
    train_graphs: list[SpatialGraph],
    config: TrainConfig,
) -> list[float]:
    """Train a model, dispatching to the correct training function."""
    if isinstance(model, SpatialTreeVAE):
        return train_spatial_vae(
            model, train_graphs,
            epochs=config.epochs, lr=config.lr,
        )
    elif isinstance(model, SpatialGraphDiffusion):
        return train_spatial_diffusion(
            model, train_graphs,
            epochs=config.epochs, lr=config.lr,
        )
    else:
        raise ValueError(f"Unknown model type: {type(model).__name__}")


def run_benchmark(
    models: dict[str, nn.Module],
    data_config: DataConfig,
    train_config: TrainConfig,
    num_eval_samples: int = 50,
) -> list[BenchmarkResult]:
    """Main benchmark entry point.

    Generates data, trains each model, evaluates, and returns results.

    Args:
        models: mapping of name -> untrained model instance
        data_config: how to generate synthetic data
        train_config: training hyperparameters
        num_eval_samples: how many graphs to sample for evaluation

    Returns:
        list of BenchmarkResult, one per model
    """
    print(f"Generating {data_config.regime} dataset: "
          f"{data_config.num_train} train, {data_config.num_test} test")
    train_graphs, test_graphs = generate_dataset(data_config)

    results = []
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name} ({_count_parameters(model):,} params)")
        print(f"{'='*50}")

        t0 = time.time()
        losses = _train_model(model, train_graphs, train_config)
        train_time = time.time() - t0

        print(f"\nEvaluating {name} ({num_eval_samples} samples)...")
        metrics = evaluate_model(model, test_graphs, num_eval_samples)

        result = BenchmarkResult(
            model_name=name,
            metrics=metrics,
            train_losses=losses,
            train_time=train_time,
            num_params=_count_parameters(model),
        )
        results.append(result)

        print(f"  MMD = {metrics.get('mmd', 'N/A'):.6f}")
        print(f"  Train time = {train_time:.1f}s")

    return results


def compare_results(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a comparison table.

    Returns:
        Formatted string suitable for printing or logging.
    """
    if not results:
        return "No results to compare."

    # collect all metric keys
    all_keys: list[str] = []
    for r in results:
        for k in r.metrics:
            if k not in all_keys and k != 'note':
                all_keys.append(k)

    # build table
    name_w = max(20, max(len(r.model_name) for r in results) + 2)
    col_w = 16
    lines = []

    # header
    header = f"{'Model':<{name_w}}"
    for key in all_keys:
        header += f"{key:>{col_w}}"
    header += f"{'params':>{col_w}}{'time (s)':>{col_w}}"
    lines.append(header)
    lines.append('-' * len(header))

    # rows
    for r in results:
        row = f"{r.model_name:<{name_w}}"
        for key in all_keys:
            val = r.metrics.get(key)
            if val is None:
                row += f"{'--':>{col_w}}"
            elif isinstance(val, float):
                row += f"{val:>{col_w}.6f}"
            else:
                row += f"{str(val):>{col_w}}"
        row += f"{r.num_params:>{col_w},}"
        row += f"{r.train_time:>{col_w}.1f}"
        lines.append(row)

    # highlight best
    lines.append('')
    lines.append('Best (lowest) per metric:')
    for key in all_keys:
        vals = [(r.model_name, r.metrics.get(key, float('inf')))
                for r in results if isinstance(r.metrics.get(key), (int, float))]
        if vals:
            best_name, best_val = min(vals, key=lambda x: x[1])
            lines.append(f"  {key}: {best_name} ({best_val:.6f})")

    return '\n'.join(lines)


def run_baseline_comparison(
    data_config: DataConfig | None = None,
    train_config: TrainConfig | None = None,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    max_nodes: int = 64,
    num_eval_samples: int = 50,
) -> list[BenchmarkResult]:
    """Convenience function: create fresh models, train, and evaluate.

    Compares SpatialTreeVAE against SpatialGraphDiffusion on the
    specified data regime.
    """
    if data_config is None:
        data_config = DataConfig(regime='tree', num_train=100, num_test=30)
    if train_config is None:
        train_config = TrainConfig(epochs=50)

    device = data_config.device or torch.device('cpu')

    models = {
        'SpatialTreeVAE': SpatialTreeVAE(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_nodes=max_nodes,
        ).to(device),
        'SpatialDiffusion': SpatialGraphDiffusion(
            max_nodes=max_nodes,
            hidden_dim=hidden_dim,
            timesteps=50,
        ).to(device),
    }

    return run_benchmark(models, data_config, train_config, num_eval_samples)
