"""Train a GNN on the Karate Club graph and save the checkpoint."""

import sys
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from backend.core.gnn.model import NodeClassifier, train_node_classifier, generate_karate_club
from backend.config import DEVICE, CHECKPOINT_DIR, GNN_DEFAULTS


def main():
    print("Loading Karate Club graph...")
    adj, features, labels = generate_karate_club()
    adj, features, labels = adj.to(DEVICE), features.to(DEVICE), labels.to(DEVICE)

    # use a few labeled nodes per class for semi-supervised training
    train_mask = torch.zeros(34, dtype=torch.bool, device=DEVICE)
    train_mask[[0, 1, 2, 33, 32, 31]] = True  # 3 per community

    model = NodeClassifier(
        in_features=features.size(1),
        hidden=GNN_DEFAULTS["hidden_dim"],
        num_classes=2,
        n_layers=GNN_DEFAULTS["num_layers"],
        dropout=GNN_DEFAULTS["dropout"],
        layer_type="gcn",
    ).to(DEVICE)

    print(f"Training on {DEVICE} for {GNN_DEFAULTS['epochs']} epochs...")
    losses, accs = train_node_classifier(
        model, adj, features, labels, train_mask,
        lr=GNN_DEFAULTS["lr"], epochs=GNN_DEFAULTS["epochs"],
    )

    # full evaluation
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        preds = logits.argmax(dim=1)
        full_acc = (preds == labels).float().mean().item()

    print(f"Final train loss: {losses[-1]:.4f}")
    print(f"Full graph accuracy: {full_acc:.4f}")

    path = CHECKPOINT_DIR / "gnn_node_classifier.pt"
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    main()
