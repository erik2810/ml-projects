"""Hyperbolic GNN models for classification and link prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional

from .manifolds import PoincareBall, ManifoldParameter, get_device
from .layers import HyperbolicGCNLayer, HyperbolicGATLayer


# ---------------------------------------------------------------------------
# Hyperbolic GNN (node classification)
# ---------------------------------------------------------------------------

class HyperbolicGNN(nn.Module):
    """Multi-layer hyperbolic GNN for node classification.

    Encodes Euclidean features into the Poincare ball via expmap0,
    processes through multiple HyperbolicGCNLayers, then decodes
    back to Euclidean space via logmap0 for classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 7,
        num_layers: int = 2,
        c: float = 1.0,
        dropout: float = 0.1,
        use_attention: bool = False,
    ):
        super().__init__()
        self.manifold = PoincareBall(c=c)
        self.c = c
        self.num_layers = num_layers

        # Euclidean input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Hyperbolic layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_c = hidden_channels
            out_c = hidden_channels
            is_last = (i == num_layers - 1)

            if use_attention:
                self.layers.append(HyperbolicGATLayer(
                    in_c, out_c, c=c,
                    num_heads=4,
                    dropout=dropout,
                    use_activation=not is_last,
                ))
            else:
                self.layers.append(HyperbolicGCNLayer(
                    in_c, out_c, c=c,
                    use_activation=not is_last,
                    dropout=dropout,
                ))

        # Euclidean output head
        self.output_head = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: Tensor) -> Tensor:
        """Project Euclidean features into the Poincare ball."""
        h = self.input_proj(x)
        h = F.silu(h)
        return self.manifold.expmap0(h)

    def decode(self, x: Tensor) -> Tensor:
        """Project from Poincare ball to Euclidean for classification."""
        h = self.manifold.logmap0(x)
        return self.output_head(h)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x: (N, in_channels) Euclidean node features.
            adj: (N, N) dense adjacency matrix.
        Returns:
            (N, out_channels) class logits.
        """
        # Encode to Poincare ball
        h = self.encode(x)

        # Process in hyperbolic space
        for layer in self.layers:
            h = layer(h, adj)

        # Decode to Euclidean
        out = self.decode(h)
        return out

    def get_embeddings(self, x: Tensor, adj: Tensor) -> Tensor:
        """Return hyperbolic embeddings (before decoding)."""
        h = self.encode(x)
        for layer in self.layers:
            h = layer(h, adj)
        return h


# ---------------------------------------------------------------------------
# Hyperbolic Embedding (link prediction)
# ---------------------------------------------------------------------------

class HyperbolicEmbedding(nn.Module):
    """Learnable node embeddings in the Poincare ball for link prediction.

    Each node has a ManifoldParameter in the Poincare ball.
    Link prediction uses a Fermi-Dirac decoder based on hyperbolic distances.
    """

    def __init__(
        self,
        num_nodes: int,
        embed_dim: int = 16,
        c: float = 1.0,
        t: float = 1.0,
        r: float = 2.0,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.manifold = PoincareBall(c=c)
        self.c = c
        self.embed_dim = embed_dim

        # Fermi-Dirac decoder parameters
        self.t = nn.Parameter(torch.tensor(t))
        self.r = nn.Parameter(torch.tensor(r))

        # Initialize embeddings near origin
        init_data = init_scale * torch.randn(num_nodes, embed_dim, dtype=torch.float32)
        init_data = self.manifold.projx(init_data)
        self.embeddings = ManifoldParameter(init_data, manifold=self.manifold)

    def get_embeddings(self) -> Tensor:
        return self.embeddings

    def fermi_dirac_decoder(self, dist: Tensor) -> Tensor:
        """Fermi-Dirac link probability: 1 / (exp((d - r) / t) + 1)."""
        return 1.0 / (torch.exp((dist.squeeze(-1) - self.r) / self.t.clamp(min=1e-6)) + 1.0)

    def forward(self, edges: Tensor) -> Tensor:
        """
        Args:
            edges: (E, 2) edge index pairs.
        Returns:
            (E,) link probabilities.
        """
        src = self.embeddings[edges[:, 0]]
        dst = self.embeddings[edges[:, 1]]
        dist = self.manifold.dist(src, dst)
        return self.fermi_dirac_decoder(dist)

    def predict_all(self) -> Tensor:
        """Compute all pairwise link probabilities."""
        N = self.embeddings.size(0)
        emb = self.embeddings
        # Pairwise distances
        dist = self.manifold.dist(
            emb.unsqueeze(1).expand(N, N, -1),
            emb.unsqueeze(0).expand(N, N, -1),
        )
        return self.fermi_dirac_decoder(dist)

    def loss(self, pos_edges: Tensor, neg_edges: Tensor) -> Tensor:
        """Binary cross-entropy loss for link prediction."""
        pos_probs = self.forward(pos_edges)
        neg_probs = self.forward(neg_edges)

        pos_loss = -torch.log(pos_probs.clamp(min=1e-10)).mean()
        neg_loss = -torch.log((1 - neg_probs).clamp(min=1e-10)).mean()

        return pos_loss + neg_loss
