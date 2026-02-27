"""
Graph Variational Autoencoder and Discrete Graph Diffusion.

References:
    - Kipf & Welling, "Variational Graph Auto-Encoders", 2016 (arXiv:1611.07308)
    - Simonovsky & Komodakis, "GraphVAE: Towards Generation of Small Graphs
      Using Variational Autoencoders", 2018 (arXiv:1802.03480)
    - Austin et al., "Structured Denoising Diffusion Models in Discrete
      State-Spaces", 2021 — inspiration for the discrete diffusion piece
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from backend.core.graph_utils import (
    normalize_adj,
    watts_strogatz,
    barabasi_albert,
    one_hot_degree_features,
)

LATENT_DIM = 64
MAX_NODES = 20  # upper bound for generated graphs


# ---------------------------------------------------------------------------
# GCN message-passing (self-contained, no external GNN imports)
# ---------------------------------------------------------------------------

def gcn_layer(h: Tensor, adj: Tensor, weight: Tensor) -> Tensor:
    """Single GCN layer: h' = ReLU(A_norm @ h @ W).

    Implements Kipf & Welling (2017) propagation rule inline.
    adj should be raw adjacency — normalization + self-loops added here.
    """
    a_norm = normalize_adj(adj, add_self_loops=True)
    return F.relu(a_norm @ h @ weight)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class GraphEncoder(nn.Module):
    """Two-layer GCN encoder with graph-level mean readout -> (mu, logvar)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(in_dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x: Tensor, adj: Tensor) -> tuple[Tensor, Tensor]:
        # two rounds of message passing
        h = gcn_layer(x, adj, self.w1)                     # (N, hidden)
        h = gcn_layer(h, adj, self.w2)                      # (N, hidden)

        # graph-level readout (mean pooling over nodes)
        h_graph = h.mean(dim=0)                             # (hidden,)

        mu = self.fc_mu(h_graph)                            # (latent,)
        logvar = self.fc_logvar(h_graph)                    # (latent,)
        return mu, logvar


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class GraphDecoder(nn.Module):
    """MLP decoder: z -> node embeddings -> adj + node features via inner product.

    Following Simonovsky & Komodakis (2018), we decode both the adjacency
    matrix (via inner product of node embeddings) and node features.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, hidden_dim: int = 256,
                 max_nodes: int = MAX_NODES, node_feat_dim: int = 11):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feat_dim = node_feat_dim

        # z -> flat node embedding matrix
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * node_feat_dim),
        )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (adj_hat, node_features_hat), both for max_nodes."""
        node_emb = self.mlp(z).view(self.max_nodes, self.node_feat_dim)  # (N, F)

        # adjacency via inner product: A_hat = sigma(Z Z^T)
        adj_hat = torch.sigmoid(node_emb @ node_emb.t())   # (N, N)

        return adj_hat, node_emb


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class GraphVAE(nn.Module):
    """Graph VAE combining encoder + decoder with KL + BCE loss.

    Optionally uses greedy permutation matching (approximate Hungarian)
    to align generated and target graphs for loss computation.
    """

    def __init__(self, max_nodes: int = MAX_NODES, node_feat_dim: int = 11,
                 latent_dim: int = LATENT_DIM, hidden_dim: int = 128,
                 use_matching: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = GraphEncoder(node_feat_dim, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(latent_dim, hidden_dim=256,
                                    max_nodes=max_nodes, node_feat_dim=node_feat_dim)
        self.max_nodes = max_nodes
        self.use_matching = use_matching

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: z = mu + eps * sigma."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor, adj: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x, adj)
        z = self.reparameterize(mu, logvar)
        adj_hat, feat_hat = self.decoder(z)
        return adj_hat, feat_hat, mu, logvar

    # --- loss computation ---------------------------------------------------

    def loss(self, adj_hat: Tensor, adj_target: Tensor,
             mu: Tensor, logvar: Tensor, beta: float = 1.0) -> dict[str, Tensor]:
        """BCE reconstruction + KL divergence.

        Args:
            adj_hat: predicted adjacency (max_nodes, max_nodes)
            adj_target: ground truth adjacency, zero-padded to max_nodes
            mu, logvar: latent distribution parameters
            beta: KL weight (beta-VAE style)
        """
        if self.use_matching:
            adj_target = self._greedy_match(adj_hat, adj_target)

        recon = F.binary_cross_entropy(adj_hat, adj_target, reduction="mean")
        # KL(q(z|x) || p(z)), closed form for Gaussian
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total = recon + beta * kl
        return {"total": total, "recon": recon, "kl": kl}

    @staticmethod
    def _greedy_match(adj_hat: Tensor, adj_target: Tensor) -> Tensor:
        """Greedy permutation matching — approximate Hungarian.

        Finds a node permutation of adj_target that minimizes
        reconstruction error against adj_hat.
        TODO: proper Hungarian matching would improve this significantly.
        """
        n = adj_target.size(0)
        perm = list(range(n))
        # greedy: for each position, pick the unmatched target node
        # that best matches the predicted node's connectivity
        used = torch.zeros(n, dtype=torch.bool, device=adj_target.device)
        order = []
        for i in range(n):
            # score each unused target node against predicted node i
            scores = torch.full((n,), float("inf"), device=adj_target.device)
            for j in range(n):
                if not used[j]:
                    scores[j] = F.binary_cross_entropy(
                        adj_hat[i], adj_target[j], reduction="sum"
                    )
            best = scores.argmin().item()
            order.append(best)
            used[best] = True

        # permute target adjacency
        idx = torch.tensor(order, device=adj_target.device)
        return adj_target[idx][:, idx]


# ---------------------------------------------------------------------------
# Discrete Denoising Diffusion on Graphs
# ---------------------------------------------------------------------------

class GraphDenoiser(nn.Module):
    """Small GCN + MLP that predicts clean adjacency from noisy input + timestep."""

    def __init__(self, node_dim: int, hidden_dim: int = 128, max_steps: int = 100):
        super().__init__()
        # timestep embedding
        self.time_emb = nn.Embedding(max_steps, hidden_dim)

        # GCN weights (2 layers)
        self.w1 = nn.Parameter(torch.empty(node_dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(hidden_dim, hidden_dim))

        # MLP head: predict edge probability for each pair
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),  # pair concat + time
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x: Tensor, adj_noisy: Tensor, t: int) -> Tensor:
        """Predict clean adjacency from noisy graph.

        Args:
            x: node features (N, F)
            adj_noisy: noisy adjacency (N, N)
            t: integer timestep

        Returns:
            adj_pred: predicted clean adjacency (N, N)
        """
        n = x.size(0)
        device = x.device

        # message passing on noisy graph
        h = gcn_layer(x, adj_noisy, self.w1)
        h = gcn_layer(h, adj_noisy, self.w2)                # (N, hidden)

        # timestep embedding
        t_tensor = torch.tensor([t], device=device)
        t_emb = self.time_emb(t_tensor).squeeze(0)          # (hidden,)

        # predict edge for each (i, j) pair
        # expand node embeddings to all pairs
        h_i = h.unsqueeze(1).expand(n, n, -1)               # (N, N, hidden)
        h_j = h.unsqueeze(0).expand(n, n, -1)               # (N, N, hidden)
        t_expand = t_emb.unsqueeze(0).unsqueeze(0).expand(n, n, -1)

        pair_feat = torch.cat([h_i, h_j, t_expand], dim=-1) # (N, N, 2*hidden + hidden)
        adj_pred = torch.sigmoid(self.edge_mlp(pair_feat).squeeze(-1))  # (N, N)

        # enforce symmetry
        adj_pred = (adj_pred + adj_pred.t()) / 2
        return adj_pred


class DenoisingDiffusion(nn.Module):
    """Simple discrete diffusion process on graph adjacency matrices.

    Forward process: flip edges with probability beta_t at each step.
    Reverse process: learned denoiser predicts clean adjacency.

    Uses a linear noise schedule from beta_start to beta_end.
    """

    def __init__(self, max_nodes: int = MAX_NODES, hidden_dim: int = 128,
                 timesteps: int = 100, beta_start: float = 0.0001,
                 beta_end: float = 0.02):
        super().__init__()
        self.max_nodes = max_nodes
        self.max_steps = timesteps
        # node features will be one-hot degree (max_degree=10 -> 11 dims)
        node_dim = 11
        self.denoiser = GraphDenoiser(node_dim, hidden_dim, timesteps)

        # linear noise schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        self.register_buffer("betas", betas)

    def q_sample(self, adj: Tensor, t: int) -> Tensor:
        """Forward process: corrupt adj by flipping edges up to step t.

        At each step, each entry flips independently with prob beta_t.
        We apply noise cumulatively for efficiency.
        """
        adj_noisy = adj.clone()
        for step in range(t + 1):
            flip_mask = (torch.rand_like(adj_noisy) < self.betas[step]).float()
            # flip: 1->0 and 0->1 where mask is 1
            adj_noisy = adj_noisy * (1 - flip_mask) + (1 - adj_noisy) * flip_mask

        # keep symmetric and zero diagonal
        adj_noisy = torch.triu(adj_noisy, diagonal=1)
        adj_noisy = adj_noisy + adj_noisy.t()
        return adj_noisy

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Training step: sample random t, corrupt, predict clean adj, return loss."""
        t = torch.randint(0, self.max_steps, (1,)).item()
        adj_noisy = self.q_sample(adj, t)
        adj_pred = self.denoiser(x, adj_noisy, t)

        # BCE between predicted clean adj and actual clean adj
        loss = F.binary_cross_entropy(adj_pred, adj)
        return loss

    @torch.no_grad()
    def sample(self, x: Tensor | None = None,
               device: torch.device | None = None) -> Tensor:
        """Generate a graph by iteratively denoising from random adjacency.

        Starts from Bernoulli(0.5) random symmetric adjacency and
        runs the reverse process for max_steps.
        """
        if device is None:
            device = next(self.parameters()).device
        if x is None:
            # default: random one-hot features for max_nodes
            n = self.max_nodes
            x = one_hot_degree_features(
                torch.zeros(n, n, device=device), max_degree=10
            )
        n = x.size(0)

        # start from random symmetric adjacency
        adj = (torch.rand(n, n, device=device) > 0.5).float()
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()

        # reverse diffusion: step from T-1 down to 0
        for t in reversed(range(self.max_steps)):
            adj_pred = self.denoiser(x, adj, t)

            # threshold predicted probabilities to get discrete adjacency
            # add small noise for stochasticity (except at t=0)
            if t > 0:
                noise_scale = self.betas[t].sqrt()
                noise = torch.rand_like(adj_pred) * noise_scale
                adj = (adj_pred + noise > 0.5).float()
            else:
                adj = (adj_pred > 0.5).float()

            # enforce symmetry and zero diagonal
            adj = torch.triu(adj, diagonal=1)
            adj = adj + adj.t()

        return adj


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _pad_graph(adj: Tensor, features: Tensor, max_nodes: int) -> tuple[Tensor, Tensor]:
    """Zero-pad a graph's adj and features to max_nodes."""
    n = adj.size(0)
    if n >= max_nodes:
        return adj[:max_nodes, :max_nodes], features[:max_nodes]

    adj_pad = torch.zeros(max_nodes, max_nodes, device=adj.device)
    adj_pad[:n, :n] = adj

    feat_pad = torch.zeros(max_nodes, features.size(1), device=features.device)
    feat_pad[:n] = features

    return adj_pad, feat_pad


def train_vae(model: GraphVAE, graphs: list[tuple[Tensor, Tensor]],
              epochs: int = 200, lr: float = 1e-3,
              beta: float = 1.0) -> list[float]:
    """Train GraphVAE on a list of (features, adj) pairs.

    Returns list of per-epoch average losses.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, adj in graphs:
            adj_pad, feat_pad = _pad_graph(adj, features, model.max_nodes)

            adj_hat, _, mu, logvar = model(feat_pad, adj_pad)
            loss_dict = model.loss(adj_hat, adj_pad, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()

            epoch_loss += loss_dict["total"].item()

        avg = epoch_loss / max(len(graphs), 1)
        losses.append(avg)

        if (epoch + 1) % 50 == 0:
            print(f"[VAE] epoch {epoch+1}/{epochs}  loss={avg:.4f}")

    return losses


def train_diffusion(model: DenoisingDiffusion, graphs: list[tuple[Tensor, Tensor]],
                    epochs: int = 200, lr: float = 1e-3) -> list[float]:
    """Train the discrete diffusion model on a list of (features, adj) pairs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, adj in graphs:
            loss = model(features, adj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg = epoch_loss / max(len(graphs), 1)
        losses.append(avg)

        if (epoch + 1) % 50 == 0:
            print(f"[Diffusion] epoch {epoch+1}/{epochs}  loss={avg:.4f}")

    return losses


# ---------------------------------------------------------------------------
# Graph generation & latent space utilities
# ---------------------------------------------------------------------------

def generate_social_skeletons(
    num_graphs: int = 50,
    sizes: list[int] | None = None,
    device: torch.device | None = None,
) -> list[tuple[Tensor, Tensor]]:
    """Create small social-network-like graphs mixing Watts-Strogatz and BA models.

    Returns list of (features, adj) pairs. Features are one-hot degree encodings.
    """
    if sizes is None:
        sizes = [8, 10, 12, 15]

    graphs = []
    for i in range(num_graphs):
        n = sizes[i % len(sizes)]
        if i % 2 == 0:
            # small-world
            k = min(4, n - 1)
            adj = watts_strogatz(n, k=k, p=0.2, device=device)
        else:
            # scale-free
            adj = barabasi_albert(n, m=2, device=device)

        features = one_hot_degree_features(adj, max_degree=10)
        graphs.append((features, adj))

    return graphs


def interpolate_latent(
    model: GraphVAE,
    graph1: tuple[Tensor, Tensor],
    graph2: tuple[Tensor, Tensor],
    steps: int = 10,
) -> list[tuple[Tensor, Tensor]]:
    """Encode two graphs and linearly interpolate in latent space.

    Returns a list of (adj_hat, feat_hat) decoded from each interpolation point.
    """
    model.eval()
    with torch.no_grad():
        feat1, adj1 = graph1
        feat2, adj2 = graph2

        mu1, _ = model.encoder(feat1, adj1)
        mu2, _ = model.encoder(feat2, adj2)

        results = []
        for i in range(steps + 1):
            alpha = i / steps
            z = (1 - alpha) * mu1 + alpha * mu2
            adj_hat, feat_hat = model.decoder(z)
            # threshold adjacency for discrete output
            adj_discrete = (adj_hat > 0.5).float()
            results.append((adj_discrete, feat_hat))

    return results
