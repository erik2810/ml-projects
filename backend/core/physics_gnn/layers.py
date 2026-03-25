"""
Physics-informed graph neural network layers.

Each layer implements a distinct connection between differential geometry
and graph neural networks:

    CotangentConv       Aggregation with geometry-derived cotangent weights,
                        optionally refined by a learnable residual.
    DiffusionConv       Multi-scale heat kernel filters learned end-to-end.
    ReactionDiffusion   Coupled diffusion (Laplacian) + learned reaction,
                        directly modelling the Gray–Scott / Turing mechanism.
    ManifoldMessagePass Message passing on Riemannian manifolds via
                        logarithmic-map aggregation and exponential-map update.
    CurvatureAttention  Attention biased by discrete curvature estimates.

All layers accept dense adjacency and 3D positions.  For layers that need
a Laplacian or edge weights, these are computed from the geometry either
exactly (when faces are provided) or approximately (from positions + adj).

References:
    Kipf & Welling, "Semi-Supervised Classification with GCNs", ICLR 2017
    Chamberlain et al., "GRAND: Graph Neural Diffusion", ICML 2021
    Bodnar et al., "Neural Sheaf Diffusion", NeurIPS 2022
    Chami et al., "Hyperbolic Graph Convolutional Neural Networks", NeurIPS 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional

from .operators import (
    geometric_edge_weights,
    weighted_laplacian,
    symmetric_normalised_laplacian,
    cotangent_laplacian,
    discrete_curvatures,
    heat_kernel,
)


# ---------------------------------------------------------------------------
# Cotangent Convolution
# ---------------------------------------------------------------------------

class CotangentConv(nn.Module):
    """Graph convolution with geometry-derived cotangent weights.

    The aggregation kernel uses weights derived from the local geometry
    (cotangent Laplacian for meshes, Belkin–Niyogi approximation otherwise),
    with an optional learnable residual that allows the network to refine
    the physics-based weights during training.

        h_i^{(l+1)} = σ( Σ_j (w_ij + α·δw_ij) h_j^{(l)} W + b )

    where w_ij are the normalised geometry weights and δw_ij is a learned
    correction from an edge MLP conditioned on relative positions.

    When learn_residual=False, this is a pure physics-based convolution.
    When learn_residual=True, the model can discover where the physics
    prior is insufficient and learn corrections — the best of both worlds.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        learn_residual: bool = True,
        residual_scale: float = 0.1,
        edge_hidden: int = 32,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learn_residual = learn_residual
        self.residual_scale = residual_scale

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        if learn_residual:
            # Edge MLP: relative position (4 features) → scalar weight correction
            self.edge_mlp = nn.Sequential(
                nn.Linear(4, edge_hidden),
                nn.SiLU(),
                nn.Linear(edge_hidden, 1),
            )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        adj: Tensor,
        faces: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (N, in_channels) node features.
            positions: (N, 3) node coordinates.
            adj: (N, N) binary adjacency.
            faces: (F, 3) optional triangle faces for exact weights.

        Returns:
            (N, out_channels) updated features.
        """
        # Compute geometry-aware weights
        if faces is not None:
            L = cotangent_laplacian(positions, faces)
            # Extract off-diagonal weights (positive)
            W = -L.clone()
            W.fill_diagonal_(0.0)
        else:
            W = geometric_edge_weights(positions, adj, method='cotangent_approx')

        # Row-normalise for stable aggregation
        D = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W / D

        # Learnable residual correction
        if self.learn_residual:
            diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
            dist = diff.norm(dim=2, keepdim=True).clamp(min=1e-8)  # (N, N, 1)
            edge_feat = torch.cat([diff, dist], dim=2)  # (N, N, 4)
            delta_w = self.edge_mlp(edge_feat).squeeze(-1)  # (N, N)
            delta_w = delta_w * (adj > 0).float()  # mask to edges
            delta_w = delta_w - delta_w.mean()  # zero-centre correction
            W_norm = W_norm + self.residual_scale * delta_w

        # Message passing
        out = W_norm @ (x @ self.weight)

        if self.bias is not None:
            out = out + self.bias

        return out


# ---------------------------------------------------------------------------
# Diffusion Convolution (multi-scale heat kernels)
# ---------------------------------------------------------------------------

class DiffusionConv(nn.Module):
    """Multi-scale graph diffusion convolution.

    Instead of a single aggregation step, applies heat diffusion at K
    learnable time scales and combines the results.  Each scale captures
    features at a different spatial resolution:

        h^{(l+1)} = σ( Σ_k θ_k · e^{t_k L} · h^{(l)} W_k )

    The diffusion times {t_k} are initialised log-uniformly and optimised
    jointly with the network.  This is related to graph wavelet neural
    networks (Xu et al. 2019) but uses the physically meaningful heat
    kernel parameterisation.

    Connection to physics: solving ∂u/∂t = Lu is the heat equation on
    the graph.  Multi-hop message passing in GNNs IS heat diffusion;
    this layer makes the time parameter explicit and learnable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_scales: int = 4,
        t_min: float = 0.1,
        t_max: float = 10.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales

        # Learnable diffusion times (in log-space for positivity)
        log_t_init = torch.linspace(math.log(t_min), math.log(t_max), num_scales)
        self.log_t = nn.Parameter(log_t_init)

        # Per-scale linear transform
        self.weight = nn.Parameter(torch.empty(num_scales, in_channels, out_channels))
        self.scale_coeffs = nn.Parameter(torch.ones(num_scales) / num_scales)

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        for k in range(self.num_scales):
            nn.init.kaiming_uniform_(self.weight[k], a=math.sqrt(5))

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        adj: Tensor,
        faces: Optional[Tensor] = None,
        precomputed_eigen: Optional[tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """
        Args:
            x: (N, in_channels) node features.
            positions: (N, 3) node positions.
            adj: (N, N) adjacency.
            faces: optional (F, 3) for exact Laplacian.
            precomputed_eigen: (eigenvalues, eigenvectors) to reuse.

        Returns:
            (N, out_channels) updated features.
        """
        # Build Laplacian
        if faces is not None:
            L = cotangent_laplacian(positions, faces)
        else:
            W = geometric_edge_weights(positions, adj)
            L = weighted_laplacian(W)

        # Eigendecompose (cache this for efficiency in the model)
        if precomputed_eigen is not None:
            eigenvalues, eigenvectors = precomputed_eigen
        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)

        # Compute diffusion at each scale
        t = self.log_t.exp()  # (K,)
        theta = F.softmax(self.scale_coeffs, dim=0)  # (K,)

        out = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)

        for k in range(self.num_scales):
            # e^{t_k λ_i} for each eigenvalue
            exp_lambda = torch.exp(t[k] * eigenvalues)  # (N,)
            # K_t = V diag(exp) V^T
            diffused = eigenvectors @ (exp_lambda.unsqueeze(1) * (eigenvectors.T @ x))
            out = out + theta[k] * (diffused @ self.weight[k])

        if self.bias is not None:
            out = out + self.bias

        return out


# ---------------------------------------------------------------------------
# Reaction-Diffusion Layer
# ---------------------------------------------------------------------------

class ReactionDiffusionLayer(nn.Module):
    """Neural reaction-diffusion layer on graphs.

    Models each layer as one step of a reaction-diffusion system:

        h^{(l+1)} = h^{(l)} + Δt · (D_diff · L · h^{(l)} + R(h^{(l)}))

    The diffusion term (L · h) uses the physics-derived graph Laplacian,
    providing topology-aware linear smoothing.  The reaction term R(h) is
    a learnable MLP that provides the nonlinear feature transformation.

    This decomposition is not just an analogy — it IS the mechanism behind
    Turing pattern formation (Gray–Scott, FitzHugh–Nagumo), where the
    interplay of diffusion and local reaction creates spatially organised
    structure from homogeneous initial conditions.

    The integration step Δt is learnable, allowing the network to control
    how much diffusion vs reaction contributes at each layer.

    References:
        Turing, "The Chemical Basis of Morphogenesis", 1952
        Chamberlain et al., "GRAND: Graph Neural Diffusion", ICML 2021
    """

    def __init__(
        self,
        channels: int,
        reaction_hidden: int = 64,
        num_species: int = 1,
        coupled: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.num_species = num_species
        self.coupled = coupled

        # Learnable diffusion coefficients (one per species per channel)
        self.log_diffusion = nn.Parameter(
            torch.zeros(num_species, channels) - 1.0  # init ~0.37
        )

        # Learnable integration time step
        self.log_dt = nn.Parameter(torch.tensor(0.0))  # init dt=1.0

        # Reaction MLP (nonlinear, node-wise)
        if coupled and num_species > 1:
            # Coupled system: all species interact
            reaction_in = channels * num_species
        else:
            reaction_in = channels

        self.reaction = nn.Sequential(
            nn.Linear(reaction_in, reaction_hidden),
            nn.SiLU(),
            nn.Linear(reaction_hidden, reaction_hidden),
            nn.SiLU(),
            nn.Linear(reaction_hidden, channels * num_species),
        )

        # Layer normalisation for stability
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        h: Tensor | list[Tensor],
        L_norm: Tensor,
    ) -> Tensor | list[Tensor]:
        """
        Args:
            h: (N, C) features for single-species, or list of (N, C) for multi-species.
            L_norm: (N, N) normalised Laplacian (from geometric weights).

        Returns:
            Updated features, same structure as input.
        """
        dt = self.log_dt.exp()
        D = self.log_diffusion.exp()  # (S, C)

        single = isinstance(h, Tensor)
        if single:
            h = [h]

        # Diffusion: D_s * L * h_s for each species
        diffusion_terms = []
        for s, h_s in enumerate(h):
            coeff = D[min(s, D.size(0) - 1)]  # (C,)
            lap_h = L_norm @ h_s  # (N, C)
            diffusion_terms.append(coeff.unsqueeze(0) * lap_h)

        # Reaction
        if self.coupled and len(h) > 1:
            reaction_input = torch.cat(h, dim=-1)  # (N, S*C)
        else:
            reaction_input = h[0]

        reaction_out = self.reaction(reaction_input)  # (N, S*C)

        # Split reaction output per species
        C = self.channels
        reaction_terms = [reaction_out[:, s*C:(s+1)*C] for s in range(len(h))]

        # Forward Euler integration with residual connection
        h_new = []
        for s in range(len(h)):
            update = dt * (diffusion_terms[s] + reaction_terms[s])
            h_s_new = h[s] + update
            h_s_new = self.norm(h_s_new)
            h_new.append(h_s_new)

        return h_new[0] if single else h_new


# ---------------------------------------------------------------------------
# Manifold Message Passing
# ---------------------------------------------------------------------------

class ManifoldMessagePassing(nn.Module):
    """Message passing on Riemannian manifolds.

    Standard GNN message passing operates in Euclidean space.  For data
    with intrinsic manifold structure (e.g. tree-like hierarchies that
    live in hyperbolic space), Euclidean aggregation introduces distortion.

    This layer performs aggregation in the tangent space:
        1. Map neighbor features to tangent space at node i via log_i(h_j)
        2. Aggregate tangent vectors (weighted mean)
        3. Apply learnable transform in tangent space
        4. Map back to manifold via exp_i(·)

    This respects the Riemannian metric and avoids the distortion of
    naively averaging points on a curved manifold.

    Requires a manifold object implementing expmap, logmap, and projx.

    References:
        Chami et al., "Hyperbolic GCNs", NeurIPS 2019
        Bachmann et al., "Constant Curvature Graph CNs", ICML 2020
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        manifold,  # Manifold object with expmap/logmap/projx
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold = manifold

        # Tangent-space linear map (Euclidean operation in T_x M)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
    ) -> Tensor:
        """
        Args:
            x: (N, D) points on the manifold.
            adj: (N, N) adjacency (or weighted adjacency).

        Returns:
            (N, out_channels) updated manifold points.
        """
        N = x.size(0)
        M = self.manifold

        # Normalise adjacency for aggregation
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        adj_norm = adj / deg

        # For each node i, aggregate neighbor features in tangent space
        out = torch.zeros(N, self.out_channels, device=x.device, dtype=x.dtype)

        for i in range(N):
            neighbors = (adj[i] > 0).nonzero(as_tuple=True)[0]
            if len(neighbors) == 0:
                out[i] = M.projx(x[i, :self.out_channels])
                continue

            # Map neighbors to tangent space at x_i
            tangent_vecs = M.logmap(
                x[i].unsqueeze(0).expand(len(neighbors), -1),
                x[neighbors],
            )  # (K, D)

            # Weighted mean in tangent space
            weights = adj_norm[i, neighbors]  # (K,)
            mean_tangent = (weights.unsqueeze(1) * tangent_vecs).sum(dim=0)  # (D,)

            # Linear transform in tangent space
            # Pad or truncate to match weight dimensions
            if mean_tangent.size(0) < self.in_channels:
                mean_tangent = F.pad(mean_tangent, (0, self.in_channels - mean_tangent.size(0)))
            else:
                mean_tangent = mean_tangent[:self.in_channels]

            transformed = mean_tangent @ self.weight  # (out_channels,)

            if self.bias is not None:
                transformed = transformed + self.bias

            # Map back to manifold
            origin = x[i, :self.out_channels] if x.size(1) >= self.out_channels else F.pad(x[i], (0, self.out_channels - x.size(1)))
            out[i] = M.expmap(origin, M.proju(origin, transformed))

        return out


# ---------------------------------------------------------------------------
# Curvature-Biased Attention
# ---------------------------------------------------------------------------

class CurvatureAttention(nn.Module):
    """Multi-head attention with curvature-derived bias.

    Standard GAT learns attention from features alone.  This layer adds
    a geometry-derived attention bias from discrete curvature estimates:

        α_ij = softmax_j( a^T [Wh_i || Wh_j] + γ · c(i,j) )

    where c(i,j) encodes the curvature difference or edge curvature between
    nodes i and j.  High-curvature regions (sharp bends, branch points)
    receive amplified attention, reflecting their geometric significance.

    The curvature bias γ is learnable: the network discovers how much
    geometry should influence attention vs purely feature-based similarity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        curvature_features: int = 6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a_l = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_r = nn.Parameter(torch.empty(num_heads, self.head_dim))

        # Curvature bias
        self.curvature_proj = nn.Linear(curvature_features, num_heads, bias=False)
        self.curvature_gate = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_l.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_r.unsqueeze(0))

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        adj: Tensor,
        curvature_dict: Optional[dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Args:
            x: (N, in_channels) node features.
            positions: (N, 3) coordinates.
            adj: (N, N) adjacency.
            curvature_dict: precomputed curvatures (from discrete_curvatures).

        Returns:
            (N, out_channels) attended features.
        """
        N = x.size(0)
        H = self.num_heads
        D = self.head_dim

        # Project features
        h = self.W(x).view(N, H, D)  # (N, H, D)

        # Attention logits (additive decomposition)
        e_l = (h * self.a_l.unsqueeze(0)).sum(dim=-1)  # (N, H)
        e_r = (h * self.a_r.unsqueeze(0)).sum(dim=-1)  # (N, H)
        attn_logits = e_l.unsqueeze(1) + e_r.unsqueeze(0)  # (N, N, H) via broadcasting
        # attn_logits[i, j, h] = e_l[i,h] + e_r[j,h]
        attn_logits = self.leaky_relu(attn_logits)

        # Curvature bias
        if curvature_dict is not None:
            # Build pairwise curvature features
            curv_feats = self._build_curvature_features(curvature_dict, N)  # (N, 6)
            # Pairwise: |c_i - c_j| captures curvature difference
            curv_i = curv_feats.unsqueeze(1).expand(-1, N, -1)
            curv_j = curv_feats.unsqueeze(0).expand(N, -1, -1)
            curv_pair = torch.cat([
                (curv_i - curv_j).abs(),
            ], dim=-1)  # (N, N, 6)
            curv_bias = self.curvature_proj(curv_pair)  # (N, N, H)
            attn_logits = attn_logits + self.curvature_gate * curv_bias

        # Mask non-edges
        mask = (adj > 0).unsqueeze(-1).expand(-1, -1, H)
        attn_logits = attn_logits.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=1)  # (N, N, H)
        attn_weights = self.dropout(attn_weights)

        # Aggregate
        out = torch.einsum('ijh,jhd->ihd', attn_weights, h)  # (N, H, D)
        return out.reshape(N, -1)  # (N, out_channels)

    @staticmethod
    def _build_curvature_features(cd: dict[str, Tensor], N: int) -> Tensor:
        """Stack curvature quantities into a feature vector."""
        feats = []
        for key in ['mean', 'gaussian', 'principal_1', 'principal_2',
                     'shape_index', 'curvedness']:
            if key in cd:
                feats.append(cd[key].unsqueeze(1))
        if feats:
            return torch.cat(feats, dim=1)
        return torch.zeros(N, 6, device=cd.get('mean', torch.zeros(1)).device)


# ---------------------------------------------------------------------------
# Position-aware edge feature encoder (shared utility)
# ---------------------------------------------------------------------------

class GeometricEdgeEncoder(nn.Module):
    """Encode relative 3D geometry into edge features.

    Computes: [Δx, Δy, Δz, ||Δ||, Δx/||Δ||, Δy/||Δ||, Δz/||Δ||]
    and passes through a 2-layer MLP.  This is the learned analogue of
    cotangent weights: both encode local geometry into the message function.
    """

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, positions: Tensor) -> Tensor:
        """
        Args:
            positions: (N, 3) node coordinates.

        Returns:
            edge_features: (N, N, out_dim) pairwise edge features.
        """
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
        dist = diff.norm(dim=2, keepdim=True).clamp(min=1e-8)  # (N, N, 1)
        direction = diff / dist  # (N, N, 3) unit direction

        edge_input = torch.cat([diff, dist, direction], dim=2)  # (N, N, 7)
        return self.mlp(edge_input)
