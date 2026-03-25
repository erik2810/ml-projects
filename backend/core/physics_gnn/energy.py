"""
Energy functionals and physics-based regularisation.

In differential geometry, energies characterise surface properties:
    - Dirichlet energy measures smoothness (||∇f||²)
    - Total variation preserves edges (||∇f||₁)
    - Willmore energy measures bending (∫ H² dA)
    - Elastic energy penalises stretching

These serve as physics-informed regularisers for GNN outputs, encouraging
predictions that respect the underlying geometry of the domain.

References:
    Pinkall & Polthier, "Computing Discrete Minimal Surfaces", 1993
    Willmore, "Riemannian Geometry", Oxford University Press, 1993
    Rudin, Osher & Fatemi, "Nonlinear Total Variation Based Noise Removal
        Algorithms", Physica D 60, 1992
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .operators import (
    cotangent_laplacian,
    geometric_edge_weights,
    weighted_laplacian,
)


# ---------------------------------------------------------------------------
# Dirichlet energy (smoothness)
# ---------------------------------------------------------------------------

def dirichlet_energy(
    f: Tensor,
    L: Tensor,
    M: Optional[Tensor] = None,
) -> Tensor:
    """Dirichlet energy E_D(f) = f^T L f (= ∫ ||∇f||² dA in the continuous limit).

    Measures how much a signal varies across the graph.  Minimising Dirichlet
    energy produces the smoothest possible signal consistent with boundary
    conditions — this is the variational formulation of the Laplace equation.

    For GNNs, this regularises predictions to be smooth w.r.t. the graph
    topology, acting as a physics-informed alternative to L2 regularisation.

    Args:
        f: (N, C) signal on nodes (C channels).
        L: (N, N) Laplacian (positive semi-definite convention, i.e. -L_cotangent).
        M: (N,) optional mass matrix for area weighting.

    Returns:
        Scalar energy (summed over channels).
    """
    if f.dim() == 1:
        f = f.unsqueeze(1)

    # E = Σ_c f_c^T L f_c
    Lf = L @ f  # (N, C)
    if M is not None:
        Lf = Lf * M.unsqueeze(1)
    energy = (f * Lf).sum()
    return energy


def dirichlet_energy_from_positions(
    f: Tensor,
    positions: Tensor,
    adj: Tensor,
    faces: Optional[Tensor] = None,
) -> Tensor:
    """Convenience: compute Dirichlet energy directly from geometry."""
    if faces is not None:
        L = -cotangent_laplacian(positions, faces)  # flip sign to positive semi-def
    else:
        W = geometric_edge_weights(positions, adj)
        L = -weighted_laplacian(W)
    return dirichlet_energy(f, L)


# ---------------------------------------------------------------------------
# Total variation (edge-preserving smoothness)
# ---------------------------------------------------------------------------

def total_variation(
    f: Tensor,
    positions: Tensor,
    adj: Tensor,
    p: float = 1.0,
    epsilon: float = 1e-8,
) -> Tensor:
    """Anisotropic total variation TV_p(f) = Σ_{(i,j)} ||f_i - f_j||^p.

    For p=1, this is the graph total variation — it encourages piecewise
    constant predictions while preserving sharp transitions (edges), unlike
    Dirichlet energy which penalises all variation equally.

    For p=2, this reduces to Dirichlet energy (up to constants).

    Args:
        f: (N, C) signal.
        positions: (N, 3) coordinates.
        adj: (N, N) adjacency.
        p: exponent (1 for L1-TV, 2 for Dirichlet).
        epsilon: stability constant for p < 2.

    Returns:
        Scalar total variation.
    """
    if f.dim() == 1:
        f = f.unsqueeze(1)

    # Compute differences along edges
    edges = adj.nonzero(as_tuple=False)  # (E, 2)
    diff = f[edges[:, 0]] - f[edges[:, 1]]  # (E, C)
    norms = diff.norm(dim=1)  # (E,)

    if p == 1:
        tv = norms.sum()
    elif p == 2:
        tv = norms.pow(2).sum()
    else:
        tv = (norms + epsilon).pow(p).sum()

    # Each edge counted twice in undirected adj
    return tv / 2.0


# ---------------------------------------------------------------------------
# Willmore energy (bending)
# ---------------------------------------------------------------------------

def willmore_energy(
    positions: Tensor,
    faces: Tensor,
) -> Tensor:
    """Willmore energy W(S) = ∫ H² dA (for surfaces only).

    Measures the bending energy of a surface.  Spheres minimise this
    among closed surfaces of given genus (Willmore conjecture, proved
    by Marques & Neves 2014).

    For GNNs generating 3D shapes, Willmore regularisation encourages
    smooth, sphere-like geometries and penalises sharp creases.

    Args:
        positions: (V, 3) vertex positions.
        faces: (F, 3) triangle faces.

    Returns:
        Scalar Willmore energy.
    """
    L, M = cotangent_laplacian(positions, faces, return_mass=True)
    M_inv = 1.0 / M.clamp(min=1e-12)

    Hn = (L @ positions) * M_inv.unsqueeze(1)  # (V, 3) mean curvature normal
    H_sq = (Hn * Hn).sum(dim=1) * 0.25  # H² per vertex

    # Integrate: Σ H²_i A_i
    return (H_sq * M).sum()


# ---------------------------------------------------------------------------
# Elastic energy (stretching penalty)
# ---------------------------------------------------------------------------

def elastic_energy(
    positions: Tensor,
    adj: Tensor,
    rest_lengths: Optional[Tensor] = None,
    stiffness: float = 1.0,
) -> Tensor:
    """Hooke's law elastic energy E = ½k Σ_{(i,j)} (||x_i-x_j|| - l_0)².

    Penalises deviations from rest lengths.  If rest_lengths is None,
    uses the mean edge length as the rest length for all edges.

    This connects GNN training to the spring-mass systems in the
    physics simulator: the same energy drives the DifferentiableForces
    optimisation and can regularise generative model outputs.

    Args:
        positions: (N, 3) node positions.
        adj: (N, N) adjacency.
        rest_lengths: (E,) per-edge rest lengths, or None for uniform.
        stiffness: spring constant k.

    Returns:
        Scalar elastic energy.
    """
    edges = adj.triu().nonzero(as_tuple=False)  # (E, 2) upper triangle
    diff = positions[edges[:, 0]] - positions[edges[:, 1]]
    lengths = diff.norm(dim=1)

    if rest_lengths is None:
        l0 = lengths.mean().detach()
    else:
        l0 = rest_lengths

    energy = 0.5 * stiffness * ((lengths - l0).pow(2)).sum()
    return energy


# ---------------------------------------------------------------------------
# Combined physics regulariser
# ---------------------------------------------------------------------------

class PhysicsRegulariser(nn.Module):
    """Combines multiple energy terms into a single differentiable loss.

    Each energy has a learnable log-weight that the optimiser can tune,
    following the multi-task learning approach of Kendall et al. (2018).
    The effective loss is:

        L_phys = Σ_k exp(-s_k) E_k + s_k

    where s_k = log(2σ_k²) is the learnable log-precision.  This
    automatically balances the scale of different energy terms.

    References:
        Kendall, Gal & Cipolla, "Multi-Task Learning Using Uncertainty
            to Weigh Losses", CVPR 2018
    """

    def __init__(
        self,
        use_dirichlet: bool = True,
        use_tv: bool = False,
        use_elastic: bool = True,
        use_willmore: bool = False,
        initial_weights: Optional[dict[str, float]] = None,
    ):
        super().__init__()
        self.terms = {}

        if use_dirichlet:
            self.terms['dirichlet'] = dirichlet_energy_from_positions
        if use_tv:
            self.terms['tv'] = total_variation
        if use_elastic:
            self.terms['elastic'] = elastic_energy

        self.use_willmore = use_willmore

        # Learnable log-precision per term
        init = initial_weights or {}
        self.log_precisions = nn.ParameterDict()
        for name in self.terms:
            val = math.log(init.get(name, 1.0)) if name in init else 0.0
            self.log_precisions[name] = nn.Parameter(torch.tensor(val))
        if use_willmore:
            val = math.log(init.get('willmore', 1.0)) if 'willmore' in init else 0.0
            self.log_precisions['willmore'] = nn.Parameter(torch.tensor(val))

    def forward(
        self,
        predictions: Tensor,
        positions: Tensor,
        adj: Tensor,
        faces: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            predictions: (N, C) GNN output to regularise.
            positions: (N, 3) node positions.
            adj: (N, N) adjacency.
            faces: optional (F, 3) for Willmore energy.

        Returns:
            Scalar physics loss.
        """
        total = torch.tensor(0.0, device=positions.device, dtype=positions.dtype)

        for name, fn in self.terms.items():
            s = self.log_precisions[name]
            precision = torch.exp(-s)

            if name == 'dirichlet':
                E = fn(predictions, positions, adj, faces)
            elif name == 'tv':
                E = fn(predictions, positions, adj)
            elif name == 'elastic':
                E = fn(positions, adj)
            else:
                E = fn(predictions, positions, adj)

            total = total + precision * E + s

        if self.use_willmore and faces is not None:
            s = self.log_precisions['willmore']
            E = willmore_energy(positions, faces)
            total = total + torch.exp(-s) * E + s

        return total
