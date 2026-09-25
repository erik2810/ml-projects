"""SE(3) equivariance of the EGNN.

A rigid motion of the input coordinates, ``x -> x R^T + t`` for a rotation
``R`` and translation ``t``, must leave the scalar node outputs unchanged
(invariance) and carry the coordinate outputs along the same motion
(equivariance). These are the defining guarantees of an E(n)-equivariant graph
network, so we check both directly on a graph mesh, to a tight tolerance.

Everything runs in float64: the equivariance is exact in real arithmetic, so the
only error is floating point, and double precision keeps it far below 1e-5.
"""

from __future__ import annotations

import torch

from backend.core.physics_gnn.layers import EGNNLayer
from backend.core.physics_gnn.models import EGNN

TOL = 1e-5


def _random_rotation(dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """A uniformly random proper rotation in SO(3), shape (3, 3)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g, dtype=dtype)
    q, r = torch.linalg.qr(a)
    # Fix the QR sign ambiguity, then force a proper rotation (det = +1).
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _grid_mesh(rows: int, cols: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """A rows x cols mesh: (N, 3) positions on a plane and (N, N) adjacency.

    Structural (axis) and diagonal edges connect each cell, and the plane is
    tilted out of the axes so the test is not accidentally 2D.
    """
    ys, xs = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    flat = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.zeros(rows * cols)], dim=1)
    positions = flat.to(dtype)

    n = rows * cols
    adj = torch.zeros(n, n, dtype=dtype)

    def idx(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    a, b = idx(r, c), idx(nr, nc)
                    adj[a, b] = 1.0
                    adj[b, a] = 1.0
    return positions, adj


def _graph(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random invariant node features, mesh positions, and adjacency."""
    torch.manual_seed(0)
    positions, adj = _grid_mesh(4, 5, dtype)
    h = torch.randn(positions.size(0), 6, dtype=dtype)  # scalar, coordinate-independent
    return h, positions, adj


def _activate(module: torch.nn.Module) -> torch.nn.Module:
    """Perturb every parameter and switch to eval.

    The coordinate readout is zero-initialised for training stability, so a fresh
    EGNN is the identity on positions, which would make equivariance hold
    trivially. Nudging all parameters makes the coordinate update genuinely
    non-trivial, so the equivariance assertions test a real map.
    """
    torch.manual_seed(7)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.2 * torch.randn_like(p))
    return module.eval()


def test_egnn_model_se3_equivariance() -> None:
    dtype = torch.float64
    h, x, adj = _graph(dtype)
    model = _activate(
        EGNN(in_channels=6, hidden_channels=16, out_channels=4, num_layers=4).to(dtype)
    )

    r = _random_rotation(dtype, seed=1)
    t = torch.randn(3, dtype=dtype)

    out_h, out_x = model(h, x, adj)
    out_h_t, out_x_t = model(h, x @ r.T + t, adj)

    # Features are invariant to the rigid motion.
    assert torch.allclose(out_h, out_h_t, atol=TOL), (out_h - out_h_t).abs().max().item()
    # Coordinates transform equivariantly: f(Rx + t) = R f(x) + t.
    assert torch.allclose(out_x @ r.T + t, out_x_t, atol=TOL), (
        (out_x @ r.T + t - out_x_t).abs().max().item()
    )
    # The coordinate update is non-trivial (not an identity that is trivially equivariant).
    assert not torch.allclose(out_x, x, atol=1e-3)


def test_egnn_layer_se3_equivariance() -> None:
    dtype = torch.float64
    h, x, adj = _graph(dtype)
    layer = _activate(EGNNLayer(feature_dim=6).to(dtype))

    r = _random_rotation(dtype, seed=2)
    t = torch.randn(3, dtype=dtype)

    h_out, x_out = layer(h, x, adj)
    h_out_t, x_out_t = layer(h, x @ r.T + t, adj)

    assert torch.allclose(h_out, h_out_t, atol=TOL)
    assert torch.allclose(x_out @ r.T + t, x_out_t, atol=TOL)
    assert not torch.allclose(x_out, x, atol=1e-3)


def test_pure_rotation_and_pure_translation() -> None:
    # Each half of the symmetry on its own, as a sharper localisation if the
    # combined test ever fails.
    dtype = torch.float64
    h, x, adj = _graph(dtype)
    model = _activate(
        EGNN(in_channels=6, hidden_channels=16, out_channels=4, num_layers=3).to(dtype)
    )
    out_h, out_x = model(h, x, adj)

    r = _random_rotation(dtype, seed=3)
    out_h_r, out_x_r = model(h, x @ r.T, adj)
    assert torch.allclose(out_h, out_h_r, atol=TOL)
    assert torch.allclose(out_x @ r.T, out_x_r, atol=TOL)

    t = torch.randn(3, dtype=dtype)
    out_h_t, out_x_t = model(h, x + t, adj)
    assert torch.allclose(out_h, out_h_t, atol=TOL)
    assert torch.allclose(out_x + t, out_x_t, atol=TOL)
