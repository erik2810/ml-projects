"""
Conditional Graph Generator — VAE that generates small graphs conditioned on
target structural properties (num_nodes, edge density, clustering coefficient).

Follows Simonovsky & Komodakis, 2018 for the graph VAE idea, but simplified:
we flatten the upper triangle of the adjacency matrix and treat it as a
fixed-length binary vector. Condition vector gets concatenated at both
encoder and decoder, similar to CVAE (Sohn et al., 2015).

Fixed graph size (max_nodes) — smaller graphs are zero-padded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from backend.core.graph_utils import (
    erdos_renyi,
    barabasi_albert,
    graph_density,
    clustering_coefficient,
)

# ---------- helpers ----------

MAX_NODES = 20
LATENT_DIM = 32
COND_DIM = 3  # [num_nodes_norm, density, clustering]


def _triu_size(n: int) -> int:
    """Number of entries in strict upper triangle of n x n matrix."""
    return n * (n - 1) // 2


def _adj_to_triu_vec(adj: Tensor) -> Tensor:
    """Flatten upper triangle of adjacency to vector. Batch-friendly: (B, N, N) -> (B, D)."""
    idx = torch.triu_indices(adj.size(-2), adj.size(-1), offset=1)
    if adj.dim() == 2:
        return adj[idx[0], idx[1]]
    return adj[:, idx[0], idx[1]]


def _triu_vec_to_adj(vec: Tensor, n: int) -> Tensor:
    """Reconstruct symmetric adj from upper-triangle vector. (B, D) -> (B, N, N)."""
    idx = torch.triu_indices(n, n, offset=1)
    if vec.dim() == 1:
        adj = torch.zeros(n, n, device=vec.device)
        adj[idx[0], idx[1]] = vec
        adj = adj + adj.t()
        return adj
    B = vec.size(0)
    adj = torch.zeros(B, n, n, device=vec.device)
    adj[:, idx[0], idx[1]] = vec
    adj = adj + adj.transpose(1, 2)
    return adj


# ---------- model ----------

class Encoder(nn.Module):
    """MLP encoder: (triu_vec, cond) -> (mu, logvar) in latent space."""

    def __init__(self, input_dim: int, cond_dim: int = COND_DIM,
                 hidden: int = 256, latent: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)

    def forward(self, x: Tensor, cond: Tensor):
        h = self.net(torch.cat([x, cond], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """MLP decoder: (z, cond) -> reconstructed triu_vec (logits before sigmoid)."""

    def __init__(self, output_dim: int, cond_dim: int = COND_DIM,
                 hidden: int = 256, latent: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            # no sigmoid here — we use BCE with logits for numerical stability
        )

    def forward(self, z: Tensor, cond: Tensor):
        return self.net(torch.cat([z, cond], dim=-1))


class ConditionalGraphGenerator(nn.Module):
    """
    Conditional VAE for graph generation.

    Input conditions: [target_num_nodes / max_nodes, target_density, target_clustering]
    All in [0, 1].
    """

    def __init__(self, max_nodes: int = MAX_NODES, latent_dim: int = LATENT_DIM,
                 hidden_dim: int = 256, beta: float = 1.0):
        super().__init__()
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim
        self.beta = beta  # KL weight, can anneal during training

        triu_dim = _triu_size(max_nodes)
        self.encoder = Encoder(triu_dim, COND_DIM, hidden_dim, latent_dim)
        self.decoder = Decoder(triu_dim, COND_DIM, hidden_dim, latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # standard reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, adj_vec: Tensor, cond: Tensor):
        mu, logvar = self.encoder(adj_vec, cond)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decoder(z, cond)
        return recon_logits, mu, logvar

    def loss(self, adj_vec: Tensor, cond: Tensor):
        """ELBO loss = reconstruction (BCE) + beta * KL divergence."""
        recon_logits, mu, logvar = self.forward(adj_vec, cond)
        # BCE on the flattened triu entries
        recon_loss = F.binary_cross_entropy_with_logits(
            recon_logits, adj_vec, reduction="mean"
        )
        # closed-form KL for diagonal Gaussian vs N(0, I)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl, recon_loss, kl

    def decode(self, z: Tensor, cond: Tensor) -> Tensor:
        """Decode latent z into adjacency prob matrix (B, N, N)."""
        logits = self.decoder(z, cond)
        probs = torch.sigmoid(logits)
        return _triu_vec_to_adj(probs, self.max_nodes)


# ---------- sampling ----------

@torch.no_grad()
def sample_graphs(model: ConditionalGraphGenerator, conditions: Tensor,
                  num_samples: int = 1, threshold: float = 0.5) -> Tensor:
    """Generate graphs from condition vectors by sampling the prior.

    Args:
        model: trained ConditionalGraphGenerator
        conditions: (C, 3) condition vectors, each row = [nodes_norm, density, clustering]
        num_samples: how many samples per condition
        threshold: binarize edge probabilities

    Returns:
        (C * num_samples, max_nodes, max_nodes) adjacency tensors
    """
    model.eval()
    device = next(model.parameters()).device

    # repeat conditions for num_samples
    cond = conditions.to(device)
    cond = cond.repeat_interleave(num_samples, dim=0)  # (C*S, 3)

    z = torch.randn(cond.size(0), model.latent_dim, device=device)
    adj_probs = model.decode(z, cond)
    # binarize
    adj = (adj_probs > threshold).float()
    return adj


# ---------- dataset construction ----------

def _pad_adj(adj: Tensor, max_nodes: int) -> Tensor:
    """Zero-pad adjacency to (max_nodes, max_nodes)."""
    n = adj.size(0)
    if n >= max_nodes:
        return adj[:max_nodes, :max_nodes]
    padded = torch.zeros(max_nodes, max_nodes, device=adj.device)
    padded[:n, :n] = adj
    return padded


def _compute_conditions(adj: Tensor, max_nodes: int) -> Tensor:
    """Compute normalized condition vector for a single graph."""
    n = adj.size(0)
    density = graph_density(adj)
    # mean clustering coeff, handle empty graphs
    cc = clustering_coefficient(adj)
    mean_cc = cc.mean().item() if n > 0 else 0.0
    # normalize node count to [0, 1]
    nodes_norm = n / max_nodes
    return torch.tensor([nodes_norm, density, mean_cc], dtype=torch.float32)


def build_training_set(num_graphs: int = 2000, max_nodes: int = MAX_NODES):
    """Generate a mix of ER and BA graphs, compute properties, return tensors.

    Returns:
        adj_vecs: (num_graphs, triu_dim) flattened upper triangles
        conds:    (num_graphs, 3) condition vectors
    """
    adj_vecs = []
    conds = []

    for i in range(num_graphs):
        # random graph size between 5 and max_nodes
        n = torch.randint(5, max_nodes + 1, (1,)).item()

        if i % 2 == 0:
            # Erdos-Renyi with random edge probability
            p = torch.empty(1).uniform_(0.1, 0.6).item()
            adj = erdos_renyi(n, p)
        else:
            # Barabasi-Albert with random attachment param
            m = torch.randint(1, min(4, n), (1,)).item()
            adj = barabasi_albert(n, m)

        cond = _compute_conditions(adj, max_nodes)
        padded = _pad_adj(adj, max_nodes)
        vec = _adj_to_triu_vec(padded)

        adj_vecs.append(vec)
        conds.append(cond)

    return torch.stack(adj_vecs), torch.stack(conds)


# ---------- training ----------

def train_generator(model: ConditionalGraphGenerator, dataset,
                    epochs: int = 100, lr: float = 1e-3, batch_size: int = 64,
                    beta_warmup: int = 10, verbose: bool = True) -> list[float]:
    """Train the CVAE with optional KL annealing (linear warmup over beta_warmup epochs).

    Following Bowman et al., 2016 — KL annealing helps avoid posterior collapse.
    Returns per-epoch average losses.
    """
    device = next(model.parameters()).device

    # accept either TensorDataset or (adj_vecs, conds) tuple
    if isinstance(dataset, tuple):
        dataset = TensorDataset(*dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    base_beta = model.beta
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        if epoch < beta_warmup:
            model.beta = base_beta * (epoch + 1) / beta_warmup
        else:
            model.beta = base_beta

        total_loss = 0.0
        n_batches = 0

        for (adj_vec_batch, cond_batch) in loader:
            adj_vec_batch = adj_vec_batch.to(device)
            cond_batch = cond_batch.to(device)

            loss, recon, kl = model.loss(adj_vec_batch, cond_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"[epoch {epoch+1:3d}/{epochs}] loss={avg_loss:.4f}  beta={model.beta:.3f}")

    model.beta = base_beta
    return epoch_losses
