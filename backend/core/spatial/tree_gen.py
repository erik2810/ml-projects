"""
Autoregressive Spatial Tree Generator.

Generates trees in 3D by sequentially adding nodes. At each step the model
decides:
    1. Which existing node to attach to (discrete — attention over existing nodes)
    2. The 3D offset from the chosen parent (continuous — Gaussian with learned μ, σ)
    3. Whether to stop (Bernoulli)

The key challenge from the PhD description: "the combination of discrete
decisions for the structure together with continuous positions in 3D space."

This is a variational model — an encoder maps full trees to latent codes,
and the decoder generates autoregressively conditioned on z. This allows
both generation from prior samples AND encoding/interpolation of real
morphologies.

Architecture is loosely inspired by:
    - Li et al., "Learning to Generate 3D Shapes" (graph-RNN style)
    - Liao et al., "Efficient Graph Generation with Graph Recurrent
      Attention Networks", NeurIPS 2019
    - GraphRNN (You et al., 2018) extended to 3D positions

But adapted for *trees* specifically — we exploit the fact that every
node has exactly one parent, which simplifies the generation compared
to general graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

from backend.core.graph_utils import normalize_adj
from .graph3d import SpatialGraph


# ---------------------------------------------------------------------------
# Encoder: spatial graph → latent code
# ---------------------------------------------------------------------------

class SpatialGraphEncoder(nn.Module):
    """Encodes a spatial tree into a latent vector.

    Uses a position-aware GNN: messages incorporate both node features
    and relative 3D positions between connected nodes. This gives the
    model access to both topology and geometry.
    """

    def __init__(self, pos_dim: int = 3, hidden_dim: int = 64,
                 latent_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.pos_encoder = nn.Linear(pos_dim, hidden_dim)
        self.edge_encoder = nn.Linear(pos_dim + 1, hidden_dim)  # relative pos + distance

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, graph: SpatialGraph) -> tuple[Tensor, Tensor]:
        n = graph.num_nodes
        if n == 0:
            d = self.fc_mu.out_features
            return torch.zeros(d), torch.zeros(d)

        # initial node embeddings from positions
        h = self.pos_encoder(graph.pos)  # (N, hidden)

        # message passing with edge features (relative position)
        a_norm = normalize_adj(graph.adj, add_self_loops=True)

        for layer in self.gnn_layers:
            # compute edge features for aggregation
            # simple version: multiply by normalized adjacency
            msg = a_norm @ h  # aggregated neighbor features
            h = F.relu(layer(torch.cat([h, msg], dim=-1)))

        # readout: mean pooling
        h_graph = h.mean(dim=0)

        return self.fc_mu(h_graph), self.fc_logvar(h_graph)


# ---------------------------------------------------------------------------
# Decoder: latent code → tree (autoregressive)
# ---------------------------------------------------------------------------

class TreeDecoder(nn.Module):
    """Autoregressively generates a spatial tree from a latent code.

    At each step t:
        - GRU state summarizes the tree built so far
        - Attention over existing nodes selects the parent
        - MLP predicts 3D position offset from parent
        - Sigmoid gate predicts stopping probability
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64,
                 pos_dim: int = 3, max_nodes: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        # latent → initial GRU state
        self.z_to_h = nn.Linear(latent_dim, hidden_dim)

        # node state encoder (position + how many children so far)
        self.node_encoder = nn.Linear(pos_dim + 1, hidden_dim)

        # GRU for maintaining generation state
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # attention: which node to attach to
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim)

        # position prediction: offset from parent
        self.pos_mu = nn.Linear(hidden_dim + hidden_dim, pos_dim)
        self.pos_logvar = nn.Linear(hidden_dim + hidden_dim, pos_dim)

        # stop prediction
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, z: Tensor, target: SpatialGraph | None = None,
                teacher_forcing: float = 0.5) -> dict:
        """Generate or reconstruct a tree.

        If target is provided, uses teacher forcing with given probability.
        Returns dict with losses (training) or generated graph (inference).
        """
        device = z.device
        h = torch.tanh(self.z_to_h(z))  # GRU hidden state

        # always start with root at origin
        positions = [torch.zeros(3, device=device)]
        parents = [-1]
        child_counts = [0]
        node_embeds = [self.node_encoder(
            torch.cat([positions[0], torch.tensor([0.0], device=device)])
        )]

        losses = {'parent': 0.0, 'pos': 0.0, 'stop': 0.0}
        n_steps = 0

        target_n = target.num_nodes if target is not None else self.max_nodes

        for t in range(1, target_n):
            # update GRU state
            # input: embedding of the most recently added node
            gru_input = node_embeds[-1]
            h = self.gru(gru_input, h)

            # === parent selection via attention ===
            query = self.attn_query(h).unsqueeze(0)  # (1, hidden)
            keys = self.attn_key(torch.stack(node_embeds))  # (t, hidden)
            attn_logits = (query @ keys.t()).squeeze(0) / math.sqrt(self.hidden_dim)
            parent_probs = F.softmax(attn_logits, dim=0)

            if target is not None and torch.rand(1).item() < teacher_forcing:
                chosen_parent = int(target.parent[t].item())
                if chosen_parent < 0:
                    chosen_parent = 0
            else:
                chosen_parent = torch.multinomial(parent_probs, 1).item()

            # parent selection loss
            if target is not None:
                true_parent = int(target.parent[t].item())
                if 0 <= true_parent < len(node_embeds):
                    losses['parent'] += F.cross_entropy(
                        attn_logits.unsqueeze(0),
                        torch.tensor([true_parent], device=device),
                    )

            # === position prediction ===
            parent_embed = node_embeds[chosen_parent]
            parent_pos = positions[chosen_parent]

            pos_input = torch.cat([h, parent_embed])
            mu = self.pos_mu(pos_input)
            logvar = self.pos_logvar(pos_input)

            if target is not None and torch.rand(1).item() < teacher_forcing:
                new_pos = target.pos[t]
            else:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                offset = mu + eps * std
                new_pos = parent_pos + offset

            # position loss (Gaussian NLL)
            if target is not None:
                true_offset = target.pos[t] - parent_pos
                pos_nll = 0.5 * (logvar + (true_offset - mu) ** 2 / (logvar.exp() + 1e-8))
                losses['pos'] += pos_nll.sum()

            # === stop prediction ===
            stop_logit = self.stop_head(h).view(1)
            if target is not None:
                # target: stop=1 if this is the last node
                is_last = 1.0 if t == target_n - 1 else 0.0
                losses['stop'] += F.binary_cross_entropy_with_logits(
                    stop_logit, torch.tensor([is_last], device=device),
                )

            # bookkeeping
            positions.append(new_pos.detach())
            parents.append(chosen_parent)
            child_counts[chosen_parent] += 1
            child_counts.append(0)

            new_embed = self.node_encoder(
                torch.cat([new_pos, torch.tensor([child_counts[-1]], device=device, dtype=torch.float32)])
            )
            node_embeds.append(new_embed)
            n_steps += 1

            # check early stop during inference
            if target is None and torch.sigmoid(stop_logit).item() > 0.5 and t > 5:
                break

        # normalize losses
        if n_steps > 0:
            for k in losses:
                losses[k] = losses[k] / n_steps

        # assemble generated graph
        n = len(positions)
        pos_tensor = torch.stack(positions)
        parent_tensor = torch.tensor(parents, dtype=torch.long, device=device)
        adj = torch.zeros(n, n, device=device)
        for i in range(n):
            p = parents[i]
            if p >= 0:
                adj[i, p] = 1.0
                adj[p, i] = 1.0

        generated = SpatialGraph(pos=pos_tensor, adj=adj, parent=parent_tensor)
        return {'graph': generated, 'losses': losses}


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------

class SpatialTreeVAE(nn.Module):
    """Variational autoencoder for spatial trees.

    Encoder: spatial GNN → (mu, logvar)
    Decoder: autoregressive tree builder conditioned on z

    Loss = reconstruction (parent CE + position NLL + stop BCE) + beta * KL
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64,
                 max_nodes: int = 100, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = SpatialGraphEncoder(
            pos_dim=3, hidden_dim=hidden_dim, latent_dim=latent_dim,
        )
        self.decoder = TreeDecoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim, max_nodes=max_nodes,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, graph: SpatialGraph,
                teacher_forcing: float = 0.5) -> dict:
        mu, logvar = self.encoder(graph)
        z = self.reparameterize(mu, logvar)

        dec_out = self.decoder(z, target=graph, teacher_forcing=teacher_forcing)

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = (dec_out['losses']['parent']
                      + dec_out['losses']['pos']
                      + dec_out['losses']['stop']
                      + self.beta * kl)

        return {
            'loss': total_loss,
            'recon_loss': dec_out['losses']['parent'] + dec_out['losses']['pos'],
            'kl': kl,
            'graph': dec_out['graph'],
        }

    @torch.no_grad()
    def generate(self, num_samples: int = 1, device: torch.device | None = None) -> list[SpatialGraph]:
        """Sample trees from the prior."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        graphs = []
        for _ in range(num_samples):
            z = torch.randn(self.latent_dim, device=device)
            out = self.decoder(z)
            graphs.append(out['graph'])
        return graphs

    @torch.no_grad()
    def interpolate(self, g1: SpatialGraph, g2: SpatialGraph,
                    steps: int = 5) -> list[SpatialGraph]:
        """Interpolate between two trees in latent space."""
        self.eval()
        mu1, _ = self.encoder(g1)
        mu2, _ = self.encoder(g2)

        graphs = []
        for i in range(steps + 1):
            alpha = i / steps
            z = (1 - alpha) * mu1 + alpha * mu2
            out = self.decoder(z)
            graphs.append(out['graph'])
        return graphs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_spatial_vae(
    model: SpatialTreeVAE,
    graphs: list[SpatialGraph],
    epochs: int = 100,
    lr: float = 1e-3,
    beta_warmup: int = 10,
    teacher_forcing_decay: float = 0.995,
) -> list[float]:
    """Train the spatial tree VAE.

    Uses teacher forcing with exponential decay and KL warmup.
    Returns per-epoch average losses.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    losses = []
    base_beta = model.beta
    tf_rate = 0.8

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
            out = model(graph, teacher_forcing=tf_rate)

            optimizer.zero_grad()
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += out['loss'].item()

        avg_loss = epoch_loss / max(len(graphs), 1)
        losses.append(avg_loss)
        tf_rate *= teacher_forcing_decay

        if (epoch + 1) % 20 == 0:
            print(f"[SpatialTreeVAE] epoch {epoch+1}/{epochs}  "
                  f"loss={avg_loss:.4f}  tf={tf_rate:.3f}  beta={model.beta:.3f}")

    model.beta = base_beta
    return losses
