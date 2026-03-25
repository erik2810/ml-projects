"""
Discrete differential geometry operators for graphs and point clouds.

Provides geometry-aware alternatives to standard graph convolution weights.
For triangle meshes these are exact (cotangent Laplacian); for general
graphs and point clouds we construct approximations from local geometry.

Mathematical background:
    The cotangent Laplacian Δf(i) = Σ_j w_ij(f_j − f_i) with
    w_ij = ½(cot α_ij + cot β_ij) is the canonical discretisation of the
    Laplace–Beltrami operator on piecewise-linear surfaces (Pinkall & Polthier
    1993, Desbrun et al. 1999).  For general spatial graphs without an
    underlying triangulation we approximate these weights via local Delaunay
    constructions and kernel-based estimators.

References:
    Pinkall & Polthier, "Computing Discrete Minimal Surfaces and Their
        Conjugates", Experimental Mathematics 2(1), 1993
    Desbrun, Meyer, Schröder & Barr, "Implicit Fairing of Irregular Meshes
        using Diffusion and Curvature Flow", SIGGRAPH 1999
    Belkin & Niyogi, "Towards a Theoretical Foundation for Laplacian-Based
        Manifold Methods", JCSS 74(8), 2008
    Crane, Weischedel & Wardetzky, "Geodesics in Heat", ACM ToG 32(5), 2013
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Cotangent weights (exact, for triangle meshes)
# ---------------------------------------------------------------------------

def cotangent_laplacian(
    positions: Tensor,
    faces: Tensor,
    return_mass: bool = False,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Build the cotangent Laplacian for a triangle mesh.

    Convention: L is negative semi-definite (off-diagonal positive, diagonal
    negative), so that Δf = Lf computes the usual Laplacian.

    Args:
        positions: (V, 3) vertex coordinates.
        faces: (F, 3) triangle indices.
        return_mass: if True, also return the lumped mass matrix (V,).

    Returns:
        L: (V, V) dense Laplacian.
        M: (V,) lumped mass diagonal (only if return_mass=True).
    """
    V = positions.size(0)
    device = positions.device

    v0 = positions[faces[:, 0]]  # (F, 3)
    v1 = positions[faces[:, 1]]
    v2 = positions[faces[:, 2]]

    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2

    # Cotangent at vertex i = dot(e_from_i, e_from_i_other) / |cross|
    # At v0: angle between e01 and -e20
    cross0 = torch.linalg.cross(e01, -e20)
    cross1 = torch.linalg.cross(e12, -e01)
    cross2 = torch.linalg.cross(e20, -e12)

    area2 = cross0.norm(dim=1).clamp(min=1e-12)  # 2 * face area

    cot0 = (e01 * (-e20)).sum(dim=1) / area2  # cot(angle at v0)
    cot1 = (e12 * (-e01)).sum(dim=1) / area2
    cot2 = (e20 * (-e12)).sum(dim=1) / area2

    # Build dense Laplacian via scatter
    L = torch.zeros(V, V, device=device, dtype=positions.dtype)

    idx = faces  # (F, 3)
    # Edge (v1, v2) opposite v0: weight = 0.5 * cot0
    w0 = 0.5 * cot0
    L[idx[:, 1], idx[:, 2]] += w0
    L[idx[:, 2], idx[:, 1]] += w0

    w1 = 0.5 * cot1
    L[idx[:, 2], idx[:, 0]] += w1
    L[idx[:, 0], idx[:, 2]] += w1

    w2 = 0.5 * cot2
    L[idx[:, 0], idx[:, 1]] += w2
    L[idx[:, 1], idx[:, 0]] += w2

    # Diagonal: negative row sum (makes L negative semi-definite)
    L.diagonal().copy_(-L.sum(dim=1))

    if not return_mass:
        return L

    # Lumped mass: M_i = (1/3) Σ_{f ∈ star(i)} area(f)
    face_areas = area2 / 2.0  # (F,)
    M = torch.zeros(V, device=device, dtype=positions.dtype)
    for k in range(3):
        M.scatter_add_(0, idx[:, k], face_areas / 3.0)

    return L, M


# ---------------------------------------------------------------------------
# Approximate cotangent weights for general spatial graphs
# ---------------------------------------------------------------------------

def geometric_edge_weights(
    positions: Tensor,
    adj: Tensor,
    method: str = 'cotangent_approx',
    epsilon: float = 1e-8,
) -> Tensor:
    """Compute geometry-aware edge weights for a spatial graph.

    For graphs with 3D positions but no explicit triangulation, we estimate
    weights that approximate the cotangent Laplacian via local Voronoi duals
    or kernel-based methods.

    Args:
        positions: (N, 3) node coordinates.
        adj: (N, N) binary adjacency matrix.
        method: one of 'cotangent_approx', 'gaussian_kernel', 'inverse_distance'.
        epsilon: numerical stability constant.

    Returns:
        W: (N, N) symmetric positive edge weights (zero where adj is zero).
    """
    N = positions.size(0)
    device = positions.device

    # Pairwise displacement and distance
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
    dist = diff.norm(dim=2).clamp(min=epsilon)  # (N, N)

    if method == 'cotangent_approx':
        # For each edge (i, j), estimate the cotangent weight from the local
        # geometry.  We use the fact that for a planar Delaunay triangulation
        # the cotangent weight w_ij ∝ 1/d_ij^2 scaled by the local Voronoi
        # cell area.  This approximation was analysed by Belkin & Niyogi (2008).
        #
        # We compute: w_ij = exp(-||x_i - x_j||^2 / (4t)) / (4πt)
        # where t is the squared mean edge length (intrinsic scale).
        edge_mask = (adj > 0).float()
        edge_dists = dist * edge_mask
        mean_edge_len = edge_dists[edge_mask > 0].mean()
        t = mean_edge_len.pow(2)

        W = torch.exp(-dist.pow(2) / (4 * t + epsilon)) / (4 * math.pi * t + epsilon)
        W = W * edge_mask

        # Symmetrise (should already be symmetric if adj is)
        W = 0.5 * (W + W.T)

    elif method == 'gaussian_kernel':
        # Standard heat kernel weight: w_ij = exp(-||x_i - x_j||^2 / (2σ^2))
        edge_mask = (adj > 0).float()
        edge_dists = dist * edge_mask
        sigma = edge_dists[edge_mask > 0].median()

        W = torch.exp(-dist.pow(2) / (2 * sigma.pow(2) + epsilon))
        W = W * edge_mask
        W = 0.5 * (W + W.T)

    elif method == 'inverse_distance':
        # w_ij = 1/||x_i - x_j||, the simplest geometry-aware weight.
        W = (1.0 / dist) * (adj > 0).float()
        W = 0.5 * (W + W.T)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Zero diagonal
    W.fill_diagonal_(0.0)

    return W


def weighted_laplacian(W: Tensor) -> Tensor:
    """Build the graph Laplacian L = D − W from a weight matrix.

    Returns a negative semi-definite matrix (diagonal = -Σ_j w_ij).
    """
    D = W.sum(dim=1)
    L = -W.clone()
    L.diagonal().copy_(D)
    # Convention: Δf = -Lf for positive semi-definite, but we use
    # the convention where L itself has positive diagonal, matching
    # the standard combinatorial Laplacian.  To get the DDG convention
    # (negative semi-definite), negate:
    return -L


def symmetric_normalised_laplacian(W: Tensor, epsilon: float = 1e-8) -> Tensor:
    """L_sym = I − D^{-1/2} W D^{-1/2}.

    This is the operator that GCN (Kipf & Welling 2017) implicitly applies
    when computing D^{-1/2} A D^{-1/2} h W.  For cotangent weights this
    becomes the mass-normalised Laplacian from DDG.
    """
    D = W.sum(dim=1).clamp(min=epsilon)
    D_inv_sqrt = D.pow(-0.5)
    W_norm = D_inv_sqrt.unsqueeze(1) * W * D_inv_sqrt.unsqueeze(0)
    I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
    return I - W_norm


# ---------------------------------------------------------------------------
# Discrete curvature for graphs (not just meshes)
# ---------------------------------------------------------------------------

def discrete_curvatures(
    positions: Tensor,
    adj: Tensor,
    faces: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Compute discrete curvature estimates for each node.

    For triangle meshes (faces provided), computes exact mean and Gaussian
    curvature.  For general spatial graphs, estimates curvature from the
    local geometry via the Laplacian and angular defect heuristics.

    Returns a dict with keys:
        'mean': (N,) signed mean curvature
        'gaussian': (N,) Gaussian curvature (angle defect)
        'principal_1': (N,) larger principal curvature
        'principal_2': (N,) smaller principal curvature
        'shape_index': (N,) Koenderink shape index in [−1, 1]
        'curvedness': (N,) Koenderink curvedness ≥ 0
    """
    N = positions.size(0)
    device = positions.device
    result = {}

    if faces is not None:
        # --- Exact mesh curvature ---
        L, M = cotangent_laplacian(positions, faces, return_mass=True)
        M_inv = 1.0 / M.clamp(min=1e-12)

        # Mean curvature normal: Hn = M^{-1} L x
        Hn = (L @ positions) * M_inv.unsqueeze(1)  # (V, 3)
        H_mag = Hn.norm(dim=1)

        # Sign from vertex normals (area-weighted)
        normals = _vertex_normals_from_faces(positions, faces)
        sign = (Hn * normals).sum(dim=1).sign()
        sign[sign == 0] = 1.0
        H = H_mag * sign * 0.5  # factor 1/2 from Δx = 2Hn convention

        # Gaussian curvature via angle defect
        K = _angle_defect(positions, faces, M)

        result['mean'] = H
        result['gaussian'] = K
    else:
        # --- Approximate curvature for general graphs ---
        # Use the Laplacian characterisation: Δx ≈ 2Hn
        W = geometric_edge_weights(positions, adj, method='cotangent_approx')
        L = weighted_laplacian(W)
        degree = (adj > 0).float().sum(dim=1).clamp(min=1)

        Hn = (L @ positions) / degree.unsqueeze(1)  # (N, 3)
        H = Hn.norm(dim=1) * 0.5

        # Approximate Gaussian curvature from angular defect
        K = _approximate_angle_defect(positions, adj)

        result['mean'] = H
        result['gaussian'] = K

    # Principal curvatures from H and K: κ₁,₂ = H ± √(H² − K)
    H = result['mean']
    K = result['gaussian']
    discriminant = (H.pow(2) - K).clamp(min=0.0)
    sqrt_disc = discriminant.sqrt()
    result['principal_1'] = H + sqrt_disc
    result['principal_2'] = H - sqrt_disc

    # Shape index (Koenderink 1990): S = (2/π) arctan((κ₁+κ₂)/(κ₁−κ₂))
    ksum = result['principal_1'] + result['principal_2']
    kdiff = (result['principal_1'] - result['principal_2']).clamp(min=1e-8)
    result['shape_index'] = (2.0 / math.pi) * torch.atan2(ksum, kdiff)

    # Curvedness: C = √((κ₁² + κ₂²)/2)
    result['curvedness'] = (0.5 * (
        result['principal_1'].pow(2) + result['principal_2'].pow(2)
    )).sqrt()

    return result


# ---------------------------------------------------------------------------
# Heat kernel and multi-scale diffusion
# ---------------------------------------------------------------------------

def heat_kernel(L: Tensor, t: float | Tensor) -> Tensor:
    """Matrix exponential e^{tL} via eigendecomposition.

    For the Laplacian convention L (negative semi-definite), the heat kernel
    K_t = e^{tL} is a low-pass filter that smooths signals on the graph.
    Small t preserves high-frequency detail; large t keeps only the DC
    component.

    This is equivalent to solving ∂u/∂t = Lu for time t.

    Args:
        L: (N, N) Laplacian (negative semi-definite).
        t: diffusion time (scalar or tensor).

    Returns:
        K: (N, N) heat kernel matrix.
    """
    # Eigendecompose: L = V Λ V^T
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    # e^{tL} = V diag(e^{t λ_i}) V^T
    if isinstance(t, (int, float)):
        exp_lambda = torch.exp(t * eigenvalues)
    else:
        exp_lambda = torch.exp(t * eigenvalues)
    K = eigenvectors @ torch.diag(exp_lambda) @ eigenvectors.T
    return K


def multiscale_diffusion_filters(
    L: Tensor,
    scales: list[float] | Tensor,
) -> Tensor:
    """Compute heat diffusion filters at multiple time scales.

    Returns a bank of (N, N) filters, one per scale, that together
    capture features at different spatial resolutions.  This is the
    graph-theoretic analogue of a multi-scale wavelet decomposition.

    Related to Hammond, Vandergheynst & Gribonval, "Wavelets on Graphs
    via Spectral Graph Theory", ACHA 30(2), 2011.

    Args:
        L: (N, N) Laplacian.
        scales: list of K diffusion times.

    Returns:
        filters: (K, N, N) diffusion filter bank.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    VT = eigenvectors.T

    if isinstance(scales, list):
        scales = torch.tensor(scales, device=L.device, dtype=L.dtype)

    # (K, N) matrix of exp(t_k * λ_i)
    exp_lambdas = torch.exp(scales.unsqueeze(1) * eigenvalues.unsqueeze(0))

    # (K, N, N) filter bank
    filters = torch.einsum('ki,ji,li->kjl', exp_lambdas, eigenvectors, eigenvectors)
    return filters


# ---------------------------------------------------------------------------
# Geodesic distances via the heat method (Crane et al. 2013)
# ---------------------------------------------------------------------------

def heat_method_distances(
    positions: Tensor,
    faces: Tensor,
    sources: Tensor | list[int],
    t: Optional[float] = None,
) -> Tensor:
    """Compute geodesic distances from source vertices using the heat method.

    Three-step algorithm:
        1. Solve (M + tL)u = δ_sources  (heat diffusion)
        2. X = −∇u / ||∇u||            (normalised gradient)
        3. Solve Lφ = ∇·X              (Poisson equation)

    Args:
        positions: (V, 3) vertex coordinates.
        faces: (F, 3) triangle indices.
        sources: source vertex indices.
        t: diffusion time (default: mean_edge_length^2).

    Returns:
        distances: (V,) geodesic distances from source set.
    """
    V = positions.size(0)
    device = positions.device

    L, M = cotangent_laplacian(positions, faces, return_mass=True)

    if t is None:
        edge_lens = []
        for k in range(3):
            i, j = faces[:, k], faces[:, (k + 1) % 3]
            edge_lens.append((positions[i] - positions[j]).norm(dim=1))
        h = torch.cat(edge_lens).mean()
        t = h.pow(2).item()

    # Step 1: diffuse heat from sources
    M_diag = torch.diag(M)
    A = M_diag - t * L  # (V, V), positive definite for t > 0

    if isinstance(sources, (int, list)):
        sources = torch.tensor([sources] if isinstance(sources, int) else sources,
                               device=device)
    b = torch.zeros(V, device=device, dtype=positions.dtype)
    b[sources] = 1.0

    u = torch.linalg.solve(A, b)  # (V,)

    # Step 2: normalised gradient per face
    grad_u = _face_gradient(positions, faces, u)  # (F, 3)
    grad_norm = grad_u.norm(dim=1, keepdim=True).clamp(min=1e-12)
    X = -grad_u / grad_norm  # (F, 3)

    # Step 3: solve Poisson
    div_X = _face_divergence(positions, faces, X)  # (V,)
    # Regularise L to make it invertible (pin one value)
    L_reg = L.clone()
    L_reg[0, :] = 0
    L_reg[0, 0] = 1.0
    div_X_reg = div_X.clone()
    div_X_reg[0] = 0.0

    phi = torch.linalg.solve(L_reg, div_X_reg)
    phi = phi - phi[sources].min()  # shift so source distance = 0

    return phi


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vertex_normals_from_faces(positions: Tensor, faces: Tensor) -> Tensor:
    """Area-weighted vertex normals."""
    V = positions.size(0)
    v0, v1, v2 = positions[faces[:, 0]], positions[faces[:, 1]], positions[faces[:, 2]]
    fn = torch.linalg.cross(v1 - v0, v2 - v0)  # (F, 3), magnitude = 2*area
    normals = torch.zeros(V, 3, device=positions.device, dtype=positions.dtype)
    for k in range(3):
        normals.scatter_add_(0, faces[:, k:k+1].expand(-1, 3), fn)
    norms = normals.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return normals / norms


def _angle_defect(
    positions: Tensor, faces: Tensor, M: Tensor,
) -> Tensor:
    """Gaussian curvature via angle defect: K_i = (2π − Σθ) / A_i."""
    V = positions.size(0)
    device = positions.device

    v0, v1, v2 = positions[faces[:, 0]], positions[faces[:, 1]], positions[faces[:, 2]]

    angle_sum = torch.zeros(V, device=device, dtype=positions.dtype)
    for k in range(3):
        i = faces[:, k]
        a = positions[faces[:, (k + 1) % 3]] - positions[i]
        b = positions[faces[:, (k + 2) % 3]] - positions[i]
        cos_angle = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-12)
        angles = torch.acos(cos_angle.clamp(-1 + 1e-7, 1 - 1e-7))
        angle_sum.scatter_add_(0, i, angles)

    K = (2 * math.pi - angle_sum) / M.clamp(min=1e-12)
    return K


def _approximate_angle_defect(positions: Tensor, adj: Tensor) -> Tensor:
    """Approximate Gaussian curvature for general graphs.

    Uses the discrete Gauss map: K_i ≈ 2π − Σ_{(j,k) consecutive neighbors}
    angle(e_ij, e_ik), normalised by a local area estimate.
    """
    N = positions.size(0)
    device = positions.device
    K = torch.full((N,), 2 * math.pi, device=device, dtype=positions.dtype)

    for i in range(N):
        neighbors = (adj[i] > 0).nonzero(as_tuple=True)[0]
        if len(neighbors) < 2:
            K[i] = 0.0
            continue

        # Compute angles between consecutive neighbor directions
        dirs = positions[neighbors] - positions[i]
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)

        # Sort by angle in the best-fit plane (PCA)
        _, _, Vh = torch.linalg.svd(dirs)
        proj = dirs @ Vh[:2].T  # project to 2D
        angles = torch.atan2(proj[:, 1], proj[:, 0])
        order = angles.argsort()
        sorted_dirs = dirs[order]

        total_angle = 0.0
        for k in range(len(sorted_dirs)):
            d1 = sorted_dirs[k]
            d2 = sorted_dirs[(k + 1) % len(sorted_dirs)]
            cos_a = (d1 * d2).sum().clamp(-1 + 1e-7, 1 - 1e-7)
            total_angle += torch.acos(cos_a)

        K[i] = 2 * math.pi - total_angle

    return K


def _face_gradient(positions: Tensor, faces: Tensor, f: Tensor) -> Tensor:
    """Per-face gradient of a scalar field on a triangle mesh.

    ∇f = (1/2A) Σ_i f_i (N × e_i) where e_i is the edge opposite vertex i.
    """
    v0, v1, v2 = positions[faces[:, 0]], positions[faces[:, 1]], positions[faces[:, 2]]
    e0 = v2 - v1  # opposite v0
    e1 = v0 - v2  # opposite v1
    e2 = v1 - v0  # opposite v2

    N = torch.linalg.cross(e2, -e1)  # face normal, magnitude = 2*area
    area2 = N.norm(dim=1, keepdim=True).clamp(min=1e-12)
    N_unit = N / area2

    f0, f1, f2 = f[faces[:, 0]], f[faces[:, 1]], f[faces[:, 2]]

    grad = (
        f0.unsqueeze(1) * torch.linalg.cross(N_unit, e0)
        + f1.unsqueeze(1) * torch.linalg.cross(N_unit, e1)
        + f2.unsqueeze(1) * torch.linalg.cross(N_unit, e2)
    ) / area2

    return grad


def _face_divergence(positions: Tensor, faces: Tensor, X: Tensor) -> Tensor:
    """Integrated divergence of a face-based vector field.

    (div X)_i = ½ Σ_{f ∈ star(i)} cot(θ_jk) ⟨e_jk, X_f⟩
    """
    V = positions.size(0)
    device = positions.device

    v0 = positions[faces[:, 0]]
    v1 = positions[faces[:, 1]]
    v2 = positions[faces[:, 2]]

    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2

    # Cotangents
    cross_mag = torch.linalg.cross(e01, -e20).norm(dim=1).clamp(min=1e-12)
    cot0 = (e01 * (-e20)).sum(dim=1) / cross_mag
    cot1 = (e12 * (-e01)).sum(dim=1) / cross_mag
    cot2 = (e20 * (-e12)).sum(dim=1) / cross_mag

    div = torch.zeros(V, device=device, dtype=positions.dtype)

    # At v0: edges e01 and e20
    div.scatter_add_(0, faces[:, 0],
                     0.5 * (cot2 * (e01 * X).sum(dim=1) + cot1 * (-e20 * X).sum(dim=1)))
    div.scatter_add_(0, faces[:, 1],
                     0.5 * (cot0 * (e12 * X).sum(dim=1) + cot2 * (-e01 * X).sum(dim=1)))
    div.scatter_add_(0, faces[:, 2],
                     0.5 * (cot1 * (e20 * X).sum(dim=1) + cot0 * (-e12 * X).sum(dim=1)))

    return div
