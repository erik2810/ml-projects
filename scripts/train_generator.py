"""Train the conditional graph generator and save checkpoint."""

import sys
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from backend.core.generator.model import (
    ConditionalGraphGenerator, build_training_set, train_generator, sample_graphs,
)
from backend.core.graph_utils import graph_density, clustering_coefficient
from backend.config import DEVICE, CHECKPOINT_DIR, GENERATOR_DEFAULTS


def main():
    cfg = GENERATOR_DEFAULTS

    print(f"Building training set ({cfg['num_training_graphs']} graphs, max {cfg['max_nodes']} nodes)...")
    dataset = build_training_set(cfg["num_training_graphs"], cfg["max_nodes"])

    model = ConditionalGraphGenerator(
        max_nodes=cfg["max_nodes"],
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
    ).to(DEVICE)

    print(f"Training on {DEVICE} for {cfg['epochs']} epochs...")
    losses = train_generator(model, dataset, epochs=cfg["epochs"], lr=cfg["lr"])

    print(f"\nFinal loss: {losses[-1]:.4f}")

    # sample a few graphs to verify
    cond = torch.tensor([[0.5, 0.3, 0.3]], dtype=torch.float32)
    samples = sample_graphs(model, cond, num_samples=4)
    for i, adj in enumerate(samples):
        deg = adj.sum(dim=1)
        active = int((deg > 0).sum().item())
        density = graph_density(adj[:active, :active])
        print(f"  Sample {i}: {active} nodes, density={density:.3f}")

    path = CHECKPOINT_DIR / "graph_generator.pt"
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    main()
