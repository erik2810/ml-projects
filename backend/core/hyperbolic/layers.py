"""Hyperbolic graph neural network layers for the Poincare ball model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional

from .manifolds import PoincareBall


# ---------------------------------------------------------------------------
# Hyperbolic Message Passing (general framework)
# ---------------------------------------------------------------------------

class HyperbolicMessagePassing(nn.Module):
    """General message passing framework in hyperbolic space.

    For each node i:
        1. logmap neighbors to tangent space at i
        2. Aggregate tangent vectors (weighted sum)
        3. Apply linear transform in tangent space
        4. expmap result back to the Poincare ball
    """

    def __init__(self, in_channels: int, out_channels: int, c: float = 1.0,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold = PoincareBall(c=c)
        self.c = c

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def compute_weights(self, x: Tensor, adj: Tensor) -> Tensor:
        """Compute aggregation weights from adjacency. Override in subclasses."""
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return adj / deg

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x: (N, in_channels) points on the Poincare ball.
            adj: (N, N) dense adjacency matrix.
        Returns:
            (N, out_channels) updated points on the Poincare ball.
        """
        N = x.size(0)
        M = self.manifold

        # Compute aggregation weights
        w = self.compute_weights(x, adj)  # (N, N)

        # Log-map all pairs: tangent vectors from i to j
        # x_i: (N, 1, D) broadcast with x_j: (1, N, D)
        x_i = x.unsqueeze(1).expand(N, N, -1)  # (N, N, D)
        x_j = x.unsqueeze(0).expand(N, N, -1)  # (N, N, D)
        log_ij = M.logmap(x_i.reshape(-1, self.in_channels),
                          x_j.reshape(-1, self.in_channels))
        log_ij = log_ij.view(N, N, self.in_channels)  # (N, N, D)

        # Weighted aggregation in tangent space
        # w: (N, N) -> (N, N, 1), log_ij: (N, N, D)
        agg = (w.unsqueeze(-1) * log_ij).sum(dim=1)  # (N, D)

        # Linear transform in tangent space
        if agg.size(-1) < self.in_channels:
            agg = F.pad(agg, (0, self.in_channels - agg.size(-1)))
        elif agg.size(-1) > self.in_channels:
            agg = agg[..., :self.in_channels]

        transformed = agg @ self.weight  # (N, out_channels)

        if self.bias is not None:
            transformed = transformed + self.bias

        # Expmap back from origin (since aggregated tangent vectors
        # approximate tangent at origin after mean-centering)
        origin = torch.zeros(N, self.out_channels, device=x.device, dtype=x.dtype)
        out = M.expmap(origin, transformed)

        return out


# ---------------------------------------------------------------------------
# Hyperbolic GCN Layer
# ---------------------------------------------------------------------------

class HyperbolicGCNLayer(nn.Module):
    """Graph convolution in the Poincare ball.

    Message passing via logmap -> aggregate -> linear -> expmap:
        1. Map features to tangent space at origin (logmap0)
        2. Aggregate with degree-normalized adjacency
        3. Linear transform
        4. Map back to Poincare ball (expmap0)
    """

    def __init__(self, in_channels: int, out_channels: int, c: float = 1.0,
                 use_activation: bool = True, dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold = PoincareBall(c=c)
        self.c = c
        self.use_activation = use_activation

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x: (N, in_channels) points on the Poincare ball.
            adj: (N, N) dense adjacency matrix.
        Returns:
            (N, out_channels) updated points on the Poincare ball.
        """
        M = self.manifold

        # 1. Map to tangent space at origin
        x_tan = M.logmap0(x)  # (N, in_channels)

        # 2. Linear transform in tangent space
        x_tan = self.dropout(x_tan)
        h = x_tan @ self.weight  # (N, out_channels)

        if self.bias is not None:
            h = h + self.bias

        # 3. Neighborhood aggregation (degree-normalized)
        # Add self-loops
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        deg = adj_hat.sum(dim=1).clamp(min=1e-8)
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = deg_inv_sqrt.unsqueeze(1) * adj_hat * deg_inv_sqrt.unsqueeze(0)
        h = adj_norm @ h  # (N, out_channels)

        # 4. Optional activation in tangent space
        if self.use_activation:
            h = F.silu(h)

        # 5. Map back to Poincare ball
        out = M.expmap0(h)

        return out


# ---------------------------------------------------------------------------
# Hyperbolic GAT Layer
# ---------------------------------------------------------------------------

class HyperbolicGATLayer(nn.Module):
    """Attention-weighted message passing in hyperbolic space.

    Computes attention in the tangent space at the origin, then
    aggregates and maps back to the Poincare ball.
    """

    def __init__(self, in_channels: int, out_channels: int, c: float = 1.0,
                 num_heads: int = 4, dropout: float = 0.0,
                 use_activation: bool = True, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold = PoincareBall(c=c)
        self.c = c
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.use_activation = use_activation

        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a_l = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_r = nn.Parameter(torch.empty(num_heads, self.head_dim))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_l.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_r.unsqueeze(0))

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x: (N, in_channels) points on the Poincare ball.
            adj: (N, N) dense adjacency matrix.
        Returns:
            (N, out_channels) updated points on the Poincare ball.
        """
        N = x.size(0)
        H = self.num_heads
        D = self.head_dim
        M = self.manifold

        # Map to tangent space at origin
        x_tan = M.logmap0(x)  # (N, in_channels)

        # Linear projection
        h = self.W(x_tan).view(N, H, D)  # (N, H, D)

        # Attention logits
        e_l = (h * self.a_l.unsqueeze(0)).sum(dim=-1)  # (N, H)
        e_r = (h * self.a_r.unsqueeze(0)).sum(dim=-1)  # (N, H)
        attn_logits = e_l.unsqueeze(1) + e_r.unsqueeze(0)  # (N, N, H)
        attn_logits = self.leaky_relu(attn_logits)

        # Mask non-edges (include self-loops)
        adj_hat = adj + torch.eye(N, device=adj.device, dtype=adj.dtype)
        mask = (adj_hat > 0).unsqueeze(-1).expand(-1, -1, H)
        attn_logits = attn_logits.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=1)  # (N, N, H)
        attn_weights = self.dropout(attn_weights)

        # Aggregate
        out = torch.einsum('ijh,jhd->ihd', attn_weights, h)  # (N, H, D)
        out = out.reshape(N, -1)  # (N, out_channels)

        if self.bias is not None:
            out = out + self.bias

        # Optional activation in tangent space
        if self.use_activation:
            out = F.silu(out)

        # Map back to Poincare ball
        out = M.expmap0(out)

        return out
