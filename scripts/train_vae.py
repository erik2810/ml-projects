"""Train the Graph VAE on social network skeletons and save checkpoint."""

import sys
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from backend.core.graphvae.model import (
    GraphVAE, train_vae, generate_social_skeletons,
)
from backend.config import DEVICE, CHECKPOINT_DIR, VAE_DEFAULTS


def main():
    cfg = VAE_DEFAULTS

    print("Generating social network skeletons...")
    graphs = generate_social_skeletons(200, [8, 10, 12, 15])
    print(f"  {len(graphs)} training graphs")

    # node feature dim = max_degree one-hot = 11 (degrees 0-10)
    node_feat_dim = graphs[0][0].size(1)

    model = GraphVAE(
        max_nodes=cfg["max_nodes"],
        node_feat_dim=node_feat_dim,
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
    ).to(DEVICE)

    print(f"Training on {DEVICE} for {cfg['epochs']} epochs...")
    losses = train_vae(model, graphs, epochs=cfg["epochs"], lr=cfg["lr"])

    print(f"\nFinal loss: {losses[-1]:.4f}")

    path = CHECKPOINT_DIR / "graph_vae.pt"
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    main()
