"""
Spatial Mesh VAE — Variational Autoencoder for general 3D graphs.

Unlike the tree VAE (tree_gen.py) which uses an autoregressive decoder
that can only produce trees, this model uses a one-shot inner-product
decoder that can reconstruct arbitrary graphs with cycles.

Architecture:
    Encoder:  SpatialGraphEncoder (reused from tree_gen.py)
              — position-aware GNN, works on any graph
    Decoder:  MeshDecoder
              z → MLP → node embeddings → inner-product adjacency
              z → MLP → 3D positions
              z → MLP → node existence mask

This is the VGAE approach (Kipf & Welling, 2016) extended to jointly
predict 3D node positions. The inner-product decoder naturally produces
symmetric adjacency matrices and can represent any graph topology.

Key capability: latent-space interpolation between meshes of different
topology (e.g., rock ↔ icosahedron), producing smooth morphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graph3d import SpatialGraph
from .tree_gen import SpatialGraphEncoder


class MeshDecoder(nn.Module):
    """One-shot decoder: latent vector → (adjacency, positions, node mask).

    For a fixed max_nodes budget:
        1. Node mask: which nodes exist (sigmoid gate)
        2. Node embeddings: per-node latent features for adjacency
        3. Adjacency: inner product of node embeddings (VGAE style)
        4. Positions: MLP predicts 3D position per node
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64,
                 max_nodes: int = 32, pos_dim: int = 3):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim

        # z → per-node features
        self.node_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes * hidden_dim),
        )

        # node existence mask
        self.mask_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes),
        )

        # position prediction
        self.pos_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pos_dim),
        )

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        """Decode latent vector to graph components.

        Returns:
            node_embeds: (max_nodes, hidden_dim)
            adj_logits:  (max_nodes, max_nodes) — symmetric
            pos:         (max_nodes, 3)
            mask_logits: (max_nodes,)
        """
        n = self.max_nodes

        # node embeddings
        node_embeds = self.node_mlp(z).view(n, self.hidden_dim)

        # adjacency via inner product (symmetric by construction)
        adj_logits = node_embeds @ node_embeds.t()

        # positions from node embeddings
        pos = self.pos_mlp(node_embeds)

        # node existence mask
        mask_logits = self.mask_mlp(z)

        return {
            'node_embeds': node_embeds,
            'adj_logits': adj_logits,
            'pos': pos,
            'mask_logits': mask_logits,
        }


class SpatialMeshVAE(nn.Module):
    """VAE for 3D meshes (general graphs with cycles).

    Encoder: SpatialGraphEncoder (position-aware GNN → mu, logvar)
    Decoder: MeshDecoder (z → adj + pos + mask, one-shot)

    Loss = adj_bce + pos_mse + mask_bce + beta * KL
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64,
                 max_nodes: int = 32, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.beta = beta

        self.encoder = SpatialGraphEncoder(
            pos_dim=3, hidden_dim=hidden_dim, latent_dim=latent_dim,
        )
        self.decoder = MeshDecoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim, max_nodes=max_nodes,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, graph: SpatialGraph) -> dict:
        """Encode graph, decode, compute loss."""
        device = graph.device
        padded = graph.pad(self.max_nodes)
        n_real = graph.num_nodes
        n_max = self.max_nodes

        # node mask target: 1 for real nodes, 0 for padding
        mask_target = torch.zeros(n_max, device=device)
        mask_target[:n_real] = 1.0

        # encode
        mu, logvar = self.encoder(graph)
        z = self.reparameterize(mu, logvar)

        # decode
        dec = self.decoder(z)

        # --- adjacency loss (BCE on upper triangle of real node pairs) ---
        pair_mask = mask_target.unsqueeze(0) * mask_target.unsqueeze(1)
        triu_mask = torch.triu(torch.ones(n_max, n_max, device=device), diagonal=1)
        pair_mask = pair_mask * triu_mask

        adj_pred = torch.sigmoid(dec['adj_logits'])
        adj_target = padded.adj

        adj_loss = F.binary_cross_entropy(
            adj_pred * pair_mask,
            adj_target * pair_mask,
            reduction='sum',
        ) / (pair_mask.sum() + 1e-8)

        # --- position loss (MSE on real nodes) ---
        pos_diff = (dec['pos'] - padded.pos) ** 2
        pos_loss = (pos_diff * mask_target.unsqueeze(1)).sum() / (n_real * 3 + 1e-8)

        # --- mask loss (BCE) ---
        mask_loss = F.binary_cross_entropy_with_logits(
            dec['mask_logits'], mask_target,
        )

        # --- KL divergence ---
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total = adj_loss + pos_loss + mask_loss + self.beta * kl

        # assemble reconstructed graph for inspection
        recon = self._decode_to_graph(dec, device)

        return {
            'loss': total,
            'adj_loss': adj_loss,
            'pos_loss': pos_loss,
            'mask_loss': mask_loss,
            'kl': kl,
            'graph': recon,
        }

    def _decode_to_graph(self, dec: dict, device: torch.device) -> SpatialGraph:
        """Convert decoder output to a SpatialGraph."""
        mask = torch.sigmoid(dec['mask_logits']) > 0.5
        n = max(mask.sum().item(), 1)

        pos = dec['pos'][:n].detach()
        adj_prob = torch.sigmoid(dec['adj_logits'][:n, :n])
        adj = (adj_prob > 0.5).float()
        # symmetrize and zero diagonal
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()

        parent = torch.full((n,), -1, dtype=torch.long, device=device)
        return SpatialGraph(pos=pos, adj=adj, parent=parent)

    @torch.no_grad()
    def generate(self, num_samples: int = 1,
                 device: torch.device | None = None) -> list[SpatialGraph]:
        """Sample meshes from the prior."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()

        graphs = []
        for _ in range(num_samples):
            z = torch.randn(self.latent_dim, device=device)
            dec = self.decoder(z)
            g = self._decode_to_graph(dec, device)
            graphs.append(g)
        return graphs

    @torch.no_grad()
    def encode(self, graph: SpatialGraph) -> Tensor:
        """Encode a graph to its latent mean (for interpolation)."""
        self.eval()
        mu, _ = self.encoder(graph)
        return mu

    @torch.no_grad()
    def interpolate(self, g1: SpatialGraph, g2: SpatialGraph,
                    steps: int = 5) -> list[SpatialGraph]:
        """Interpolate between two graphs in latent space."""
        self.eval()
        device = next(self.parameters()).device
        mu1, _ = self.encoder(g1.to(device))
        mu2, _ = self.encoder(g2.to(device))

        graphs = []
        for i in range(steps + 1):
            alpha = i / steps
            z = (1 - alpha) * mu1 + alpha * mu2
            dec = self.decoder(z)
            g = self._decode_to_graph(dec, device)
            graphs.append(g)
        return graphs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mesh_vae(
    model: SpatialMeshVAE,
    graphs: list[SpatialGraph],
    epochs: int = 100,
    lr: float = 1e-3,
    beta_warmup: int = 10,
) -> list[float]:
    """Train the mesh VAE. Returns per-epoch average losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    losses = []
    base_beta = model.beta

    for epoch in range(epochs):
        model.train()

        # KL warmup
        if epoch < beta_warmup:
            model.beta = base_beta * (epoch + 1) / beta_warmup
        else:
            model.beta = base_beta

        epoch_loss = 0.0
        for graph in graphs:
            graph = graph.to(device)
            out = model(graph)

            optimizer.zero_grad()
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += out['loss'].item()

        avg = epoch_loss / max(len(graphs), 1)
        losses.append(avg)

        if (epoch + 1) % 20 == 0:
            print(f"[MeshVAE] epoch {epoch+1}/{epochs}  "
                  f"loss={avg:.4f}  adj={out['adj_loss']:.4f}  "
                  f"pos={out['pos_loss']:.4f}  beta={model.beta:.3f}")

    model.beta = base_beta
    return losses
