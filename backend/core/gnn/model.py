"""High-level GNN models for node and graph classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.core.gnn.layers import GCNLayer, GATLayer


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class NodeClassifier(nn.Module):
    """Stack of GCN or GAT layers for semi-supervised node classification."""

    def __init__(self, in_features: int, hidden: int, num_classes: int,
                 n_layers: int = 2, dropout: float = 0.5,
                 layer_type: str = "gcn", n_heads: int = 4):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        Layer = _resolve_layer(layer_type)

        # first layer
        if layer_type == "gat":
            self.layers.append(GATLayer(in_features, hidden, n_heads=n_heads,
                                        concat=True, dropout=dropout))
            cur_dim = hidden * n_heads
        else:
            self.layers.append(Layer(in_features, hidden, dropout=dropout))
            cur_dim = hidden

        # hidden layers
        for _ in range(n_layers - 2):
            if layer_type == "gat":
                self.layers.append(GATLayer(cur_dim, hidden, n_heads=n_heads,
                                            concat=True, dropout=dropout))
                cur_dim = hidden * n_heads
            else:
                self.layers.append(Layer(cur_dim, hidden, dropout=dropout))
                cur_dim = hidden

        # output layer -- GAT uses averaging instead of concat here (Velickovic et al.)
        if layer_type == "gat":
            self.layers.append(GATLayer(cur_dim, num_classes, n_heads=1,
                                        concat=False, dropout=dropout))
        else:
            self.layers.append(Layer(cur_dim, num_classes, dropout=0.0))

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, adj))
        # no activation on final logits
        return self.layers[-1](x, adj)


class GraphClassifier(nn.Module):
    """GNN encoder + global pooling + MLP head for graph-level prediction."""

    def __init__(self, in_features: int, hidden: int, num_classes: int,
                 n_layers: int = 3, dropout: float = 0.5,
                 layer_type: str = "gcn", pool: str = "mean",
                 n_heads: int = 4):
        super().__init__()
        self.pool = pool
        self.dropout = dropout

        Layer = _resolve_layer(layer_type)
        self.convs = nn.ModuleList()

        if layer_type == "gat":
            self.convs.append(GATLayer(in_features, hidden, n_heads=n_heads,
                                       concat=True, dropout=dropout))
            cur_dim = hidden * n_heads
        else:
            self.convs.append(Layer(in_features, hidden, dropout=dropout))
            cur_dim = hidden

        for _ in range(n_layers - 1):
            if layer_type == "gat":
                self.convs.append(GATLayer(cur_dim, hidden, n_heads=n_heads,
                                           concat=True, dropout=dropout))
                cur_dim = hidden * n_heads
            else:
                self.convs.append(Layer(cur_dim, hidden, dropout=dropout))
                cur_dim = hidden

        # readout MLP -- one hidden layer is usually enough
        self.mlp = nn.Sequential(
            nn.Linear(cur_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def _readout(self, h):
        """Global pooling over the node dimension."""
        if self.pool == "sum":
            return h.sum(dim=0)
        return h.mean(dim=0)  # default: mean

    def forward(self, features_list, adj_list):
        """Classify a batch of graphs.

        Args:
            features_list: list of (N_i, F) tensors
            adj_list: list of (N_i, N_i) tensors
        Returns:
            (B, num_classes) logits
        """
        graph_embeds = []
        for x, adj in zip(features_list, adj_list):
            h = x
            for conv in self.convs:
                h = F.elu(conv(h, adj))
            graph_embeds.append(self._readout(h))

        batch = torch.stack(graph_embeds, dim=0)
        return self.mlp(batch)


def _resolve_layer(name: str):
    if name == "gat":
        return GATLayer
    return GCNLayer  # default


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_node_classifier(model, adj, features, labels, train_mask,
                          lr: float = 0.01, weight_decay: float = 5e-4,
                          epochs: int = 200) -> tuple[list[float], list[float]]:
    """Standard training loop for semi-supervised node classification."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    losses, accs = [], []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # accuracy on training nodes
        with torch.no_grad():
            preds = logits[train_mask].argmax(dim=1)
            acc = (preds == labels[train_mask]).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

    return losses, accs


# ---------------------------------------------------------------------------
# Zachary's Karate Club
# ---------------------------------------------------------------------------

def generate_karate_club() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (adj, features, labels) for the 34-node karate club graph.

    Edge list from Zachary (1977). Labels: 0 = Mr. Hi's group, 1 = Officer's group.
    """
    # 0-indexed edge list (undirected, each edge listed once)
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
        (3, 7), (3, 12), (3, 13),
        (4, 6), (4, 10),
        (5, 6), (5, 10), (5, 16),
        (6, 16),
        (8, 30), (8, 32), (8, 33),
        (9, 33),
        (13, 33),
        (14, 32), (14, 33),
        (15, 32), (15, 33),
        (18, 32), (18, 33),
        (19, 33),
        (20, 32), (20, 33),
        (22, 32), (22, 33),
        (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31),
        (25, 31),
        (26, 29), (26, 33),
        (27, 33),
        (28, 31), (28, 33),
        (29, 32), (29, 33),
        (30, 32), (30, 33),
        (31, 32), (31, 33),
        (32, 33),
    ]

    N = 34
    adj = torch.zeros(N, N)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # ground truth communities from the original study
    # 0 = Mr. Hi (node 0), 1 = Officer (node 33)
    labels = torch.tensor([
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1,
    ])

    # one-hot degree features -- simple but effective baseline encoding
    deg = adj.sum(dim=1).long()
    max_deg = deg.max().item()
    features = F.one_hot(deg, num_classes=max_deg + 1).float()

    return adj, features, labels
