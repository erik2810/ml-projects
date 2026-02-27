"""
Joint Discrete-Continuous Diffusion for Spatial Graphs.

The core idea: simultaneously denoise graph structure (discrete edges)
and node positions (continuous R^3). This is the key challenge from the
PhD description — "the combination of discrete decisions for the structure
together with continuous positions in 3D space."

Two coupled diffusion processes:
    - Edges: discrete corruption via independent bit-flips (Austin et al., 2021)
    - Positions: standard Gaussian diffusion (Ho et al., 2020)

A shared GNN backbone denoises both jointly, allowing the model to learn
the coupling between topology and geometry. Crucially, unlike molecules,
spatial proximity does NOT imply connectivity — the model must learn this
distinction.

References:
    - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
    - Austin et al., "Structured Denoising Diffusion Models in Discrete
      State-Spaces", NeurIPS 2021
    - Vignac et al., "DiGress: Discrete Denoising Diffusion for Graph
      Generation", ICLR 2023 — closest prior work, but on flat graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

from backend.core.graph_utils import normalize_adj
from .graph3d import SpatialGraph


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """Cosine schedule from Nichol & Dhariwal (2021). Smoother than linear."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(0.0001, 0.999).float()


# ---------------------------------------------------------------------------
# Denoiser network
# ---------------------------------------------------------------------------

class JointDenoiser(nn.Module):
    """GNN that predicts clean structure and positions from noisy input.

    Takes noisy (adj, positions) + timestep and outputs:
        - Edge probabilities (N, N) for the clean adjacency
        - Position offsets (N, 3) — predicts the noise (epsilon-parameterization)

    The architecture uses position-aware message passing: edge messages
    depend on both node features and relative 3D positions. This is
    essential for learning the structure-geometry coupling.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3,
                 max_nodes: int = 64, timesteps: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim

        # timestep embedding (sinusoidal, as in original DDPM)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # position → initial node features
        self.pos_encoder = nn.Linear(3, hidden_dim)

        # GNN message-passing layers
        self.mp_layers = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.mp_layers.append(nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim))
            # edge features from relative position + distance
            self.edge_mlps.append(nn.Sequential(
                nn.Linear(4, hidden_dim),  # (dx, dy, dz, dist)
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        # output heads
        self.adj_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def _sinusoidal_embedding(self, t: Tensor) -> Tensor:
        """Sinusoidal positional embedding for timestep."""
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

    def forward(self, pos_noisy: Tensor, adj_noisy: Tensor,
                t: Tensor, node_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Args:
            pos_noisy: (N, 3) noisy positions
            adj_noisy: (N, N) noisy adjacency
            t: (1,) or scalar timestep
            node_mask: (N,) binary mask for real vs padded nodes

        Returns:
            adj_pred: (N, N) predicted clean edge probabilities
            eps_pred: (N, 3) predicted noise on positions
        """
        n = pos_noisy.size(0)
        device = pos_noisy.device

        # timestep conditioning
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_emb = self._sinusoidal_embedding(t)  # (1, hidden)
        t_emb = self.time_mlp(t_emb).squeeze(0)  # (hidden,)

        # initial node embeddings
        h = self.pos_encoder(pos_noisy) + t_emb.unsqueeze(0)  # (N, hidden)

        # message passing on the noisy graph
        for mp_layer, edge_mlp in zip(self.mp_layers, self.edge_mlps):
            # compute pairwise relative positions
            rel_pos = pos_noisy.unsqueeze(1) - pos_noisy.unsqueeze(0)  # (N, N, 3)
            rel_dist = rel_pos.norm(dim=-1, keepdim=True)  # (N, N, 1)
            edge_feat = edge_mlp(torch.cat([rel_pos, rel_dist], dim=-1))  # (N, N, hidden)

            # aggregate: weighted by noisy adjacency + self-loops
            a_weighted = adj_noisy.unsqueeze(-1) * edge_feat  # mask by edges
            msg = a_weighted.sum(dim=1) + h  # self-loop

            h = F.silu(mp_layer(torch.cat([h, msg, t_emb.unsqueeze(0).expand(n, -1)], dim=-1)))

        # predict clean adjacency (edge probabilities for all pairs)
        h_i = h.unsqueeze(1).expand(n, n, -1)
        h_j = h.unsqueeze(0).expand(n, n, -1)
        adj_logits = self.adj_head(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)
        # symmetrize
        adj_logits = (adj_logits + adj_logits.t()) / 2
        adj_pred = torch.sigmoid(adj_logits)

        # predict position noise (epsilon-parameterization)
        eps_pred = self.pos_head(torch.cat([h, t_emb.unsqueeze(0).expand(n, -1)], dim=-1))

        # mask padded nodes
        if node_mask is not None:
            adj_pred = adj_pred * node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
            eps_pred = eps_pred * node_mask.unsqueeze(1)

        return adj_pred, eps_pred


# ---------------------------------------------------------------------------
# Joint diffusion process
# ---------------------------------------------------------------------------

class SpatialGraphDiffusion(nn.Module):
    """Joint discrete-continuous diffusion for spatial graph generation.

    Forward process:
        - Positions: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
        - Edges:     flip with probability beta_edge_t at each step

    Reverse process:
        - Shared denoiser predicts clean adj and position noise
        - Positions updated via DDPM reverse step
        - Edges thresholded from predicted probabilities
    """

    def __init__(self, max_nodes: int = 64, hidden_dim: int = 64,
                 timesteps: int = 100, num_layers: int = 3):
        super().__init__()
        self.max_nodes = max_nodes
        self.timesteps = timesteps

        self.denoiser = JointDenoiser(
            hidden_dim=hidden_dim, num_layers=num_layers,
            max_nodes=max_nodes, timesteps=timesteps,
        )

        # continuous noise schedule (for positions)
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', alpha_bar.sqrt())
        self.register_buffer('sqrt_one_minus_alpha_bar', (1 - alpha_bar).sqrt())

        # discrete noise schedule (for edges) — separate, typically lower noise
        edge_betas = betas * 0.3  # edges flip less aggressively
        self.register_buffer('edge_betas', edge_betas)

    def q_pos(self, x0: Tensor, t: int) -> tuple[Tensor, Tensor]:
        """Forward process for positions: add Gaussian noise."""
        eps = torch.randn_like(x0)
        xt = self.sqrt_alpha_bar[t] * x0 + self.sqrt_one_minus_alpha_bar[t] * eps
        return xt, eps

    def q_adj(self, adj: Tensor, t: int) -> Tensor:
        """Forward process for edges: flip bits with cumulative probability."""
        # cumulative flip probability up to step t
        flip_prob = 1.0 - torch.prod(1.0 - self.edge_betas[:t+1])
        flip_mask = (torch.rand_like(adj) < flip_prob).float()
        adj_noisy = adj * (1 - flip_mask) + (1 - adj) * flip_mask
        # symmetrize and zero diagonal
        adj_noisy = torch.triu(adj_noisy, diagonal=1)
        adj_noisy = adj_noisy + adj_noisy.t()
        return adj_noisy

    def forward(self, graph: SpatialGraph) -> dict[str, Tensor]:
        """Training step: sample t, corrupt, predict, compute loss."""
        device = graph.device
        padded = graph.pad(self.max_nodes)
        node_mask = torch.zeros(self.max_nodes, device=device)
        node_mask[:graph.num_nodes] = 1.0

        t = torch.randint(0, self.timesteps, (1,), device=device).item()
        t_tensor = torch.tensor([t], device=device)

        # corrupt positions
        pos_noisy, eps_true = self.q_pos(padded.pos, t)

        # corrupt edges
        adj_noisy = self.q_adj(padded.adj, t)

        # predict
        adj_pred, eps_pred = self.denoiser(pos_noisy, adj_noisy, t_tensor, node_mask)

        # === losses ===
        # position loss: MSE on predicted noise (standard DDPM)
        pos_loss = F.mse_loss(
            eps_pred * node_mask.unsqueeze(1),
            eps_true * node_mask.unsqueeze(1),
        )

        # adjacency loss: BCE on real node pairs only
        pair_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
        # zero diagonal
        pair_mask = pair_mask * (1 - torch.eye(self.max_nodes, device=device))

        adj_loss = F.binary_cross_entropy(
            adj_pred * pair_mask,
            padded.adj * pair_mask,
        )

        total = pos_loss + adj_loss

        return {
            'loss': total,
            'pos_loss': pos_loss,
            'adj_loss': adj_loss,
        }

    @torch.no_grad()
    def sample(self, num_nodes: int | None = None,
               device: torch.device | None = None) -> SpatialGraph:
        """Generate a spatial graph via reverse diffusion.

        Starts from random noise for both positions and edges,
        iteratively denoises to produce a clean graph.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        n = num_nodes or self.max_nodes
        node_mask = torch.ones(n, device=device)

        # start from noise
        pos = torch.randn(n, 3, device=device)
        adj = (torch.rand(n, n, device=device) > 0.5).float()
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t], device=device)

            adj_pred, eps_pred = self.denoiser(pos, adj, t_tensor, node_mask)

            # DDPM reverse step for positions
            if t > 0:
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alpha_bar[t]
                beta_t = self.betas[t]

                pos_pred_x0 = (pos - self.sqrt_one_minus_alpha_bar[t] * eps_pred) / self.sqrt_alpha_bar[t]
                # posterior mean
                coeff1 = beta_t * self.sqrt_alpha_bar[t-1] / (1 - alpha_bar_t)
                coeff2 = (1 - self.alpha_bar[t-1]) * alpha_t.sqrt() / (1 - alpha_bar_t)
                pos_mean = coeff1 * pos_pred_x0 + coeff2 * pos

                noise = torch.randn_like(pos) * beta_t.sqrt()
                pos = pos_mean + noise
            else:
                pos = (pos - self.sqrt_one_minus_alpha_bar[0] * eps_pred) / self.sqrt_alpha_bar[0]

            # update adjacency from prediction (with noise for t > 0)
            if t > 0:
                noise_scale = self.edge_betas[t].sqrt()
                adj = (adj_pred + torch.rand_like(adj_pred) * noise_scale > 0.5).float()
            else:
                adj = (adj_pred > 0.5).float()

            adj = torch.triu(adj, diagonal=1)
            adj = adj + adj.t()

        # build graph — try to extract tree structure
        parent = _extract_tree_parents(adj, pos)
        return SpatialGraph(pos=pos, adj=adj, parent=parent)


def _extract_tree_parents(adj: Tensor, pos: Tensor) -> Tensor:
    """Extract a spanning tree from an adjacency matrix via BFS from the
    node closest to the centroid (heuristic root selection)."""
    n = adj.size(0)
    parent = torch.full((n,), -1, dtype=torch.long, device=adj.device)

    # pick root: node closest to centroid
    centroid = pos.mean(dim=0)
    dists = torch.norm(pos - centroid.unsqueeze(0), dim=1)
    root = dists.argmin().item()

    visited = {root}
    queue = [root]
    while queue:
        node = queue.pop(0)
        neighbors = (adj[node] > 0).nonzero(as_tuple=True)[0].tolist()
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                parent[nb] = node
                queue.append(nb)

    return parent


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_spatial_diffusion(
    model: SpatialGraphDiffusion,
    graphs: list[SpatialGraph],
    epochs: int = 200,
    lr: float = 3e-4,
) -> list[float]:
    """Train the joint diffusion model. Returns per-epoch average losses."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    device = next(model.parameters()).device
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for graph in graphs:
            graph = graph.to(device)
            out = model(graph)

            optimizer.zero_grad()
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += out['loss'].item()

        avg = epoch_loss / max(len(graphs), 1)
        losses.append(avg)

        if (epoch + 1) % 25 == 0:
            print(f"[SpatialDiffusion] epoch {epoch+1}/{epochs}  "
                  f"loss={avg:.4f}  (pos={out['pos_loss']:.4f}  adj={out['adj_loss']:.4f})")

    return losses
