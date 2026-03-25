"""Hyperbolic physics simulation: spring-mass system on the Poincare disk."""

import torch
from torch import Tensor
import math
from typing import Optional, Tuple, List, Dict

from .manifolds import PoincareBall, ManifoldParameter, RiemannianAdam, get_device


# ---------------------------------------------------------------------------
# Hyperbolic Forces
# ---------------------------------------------------------------------------

class HyperbolicForces:
    """Force computations in the Poincare disk model."""

    EPSILON = 1e-6

    @staticmethod
    def spring_forces(
        positions: Tensor,
        edge_indices: Tensor,
        manifold: PoincareBall,
        spring_k: float = 2.0,
        target_length: float = 0.3,
    ) -> Tuple[Tensor, float]:
        """Compute spring forces along edges using hyperbolic distances.

        Args:
            positions: (N, 2) Poincare disk positions.
            edge_indices: (E, 2) edge connectivity.
            manifold: PoincareBall instance.
            spring_k: spring constant.
            target_length: target hyperbolic distance.
        Returns:
            forces: (N, 2) spring force per vertex.
            energy: total spring potential energy.
        """
        v1_idx = edge_indices[:, 0]
        v2_idx = edge_indices[:, 1]

        p1 = positions[v1_idx]
        p2 = positions[v2_idx]

        # Hyperbolic distances
        dist = manifold.dist(p1, p2).squeeze(-1)
        dist = dist.clamp(min=HyperbolicForces.EPSILON)

        # Direction (Euclidean approximation in tangent space)
        direction = p2 - p1
        dir_norm = direction.norm(dim=1, keepdim=True).clamp(min=HyperbolicForces.EPSILON)
        direction = direction / dir_norm

        # Spring force magnitude
        spring_mag = spring_k * (dist - target_length)

        spring_force = spring_mag.unsqueeze(1) * direction

        # Accumulate forces per vertex
        forces = torch.zeros_like(positions)
        forces.index_add_(0, v1_idx, spring_force)
        forces.index_add_(0, v2_idx, -spring_force)

        energy = 0.5 * spring_k * ((dist - target_length) ** 2).sum()

        return forces, energy.item()

    @staticmethod
    def charge_forces(
        positions: Tensor,
        charge_c: float = 0.05,
        max_vertices: int = 500,
    ) -> Tensor:
        """Compute pairwise charge repulsion forces. O(N^2).

        Args:
            positions: (N, 2) Poincare disk positions.
            charge_c: charge constant.
            max_vertices: skip if N exceeds this.
        Returns:
            forces: (N, 2) repulsion force per vertex.
        """
        n = positions.shape[0]
        forces = torch.zeros_like(positions)

        if charge_c <= 0 or n > max_vertices:
            return forces

        for i in range(n):
            diff = positions - positions[i:i+1]
            dist_sq = (diff * diff).sum(dim=1).clamp(min=1e-4)
            dist_all = torch.sqrt(dist_sq)

            repulsion = charge_c * charge_c / dist_sq.clamp(min=0.01)

            dir_away = diff / dist_all.unsqueeze(1).clamp(min=HyperbolicForces.EPSILON)

            mask = torch.ones(n, device=positions.device)
            mask[i] = 0
            forces = forces + repulsion.unsqueeze(1) * dir_away * mask.unsqueeze(1)

        return forces


# ---------------------------------------------------------------------------
# Hyperbolic Simulation
# ---------------------------------------------------------------------------

class HyperbolicSimulation:
    """Spring-mass system on the Poincare disk with Riemannian optimization."""

    def __init__(
        self,
        n_nodes: int,
        edges: List[Tuple[int, int]],
        params: Optional[Dict] = None,
        initial_positions: Optional[Tensor] = None,
    ):
        self.device = get_device()
        self.dtype = torch.float32
        self.N = n_nodes
        self.edges = edges

        defaults = {
            'spring_k': 2.0,
            'target_L': 0.5,
            'charge_c': 0.1,
            'lr': 0.03,
            'curvature': 1.0,
            'init_scale': 0.15,
            'dim': 2,
        }
        self.params = defaults
        if params is not None:
            self.params.update(params)

        self.c = self.params['curvature']
        self.dim = self.params['dim']
        self.manifold = PoincareBall(c=self.c)

        self._init_state(initial_positions)

    def _init_state(self, initial_positions: Optional[Tensor] = None):
        init_scale = self.params['init_scale']

        if initial_positions is not None:
            x0 = initial_positions.to(self.dtype).to(self.device)
            x0 = self.manifold.projx(x0, eps=0.1)
        else:
            x0 = init_scale * torch.randn(
                self.N, self.dim, dtype=self.dtype, device=self.device
            )

        self.X = ManifoldParameter(x0, manifold=self.manifold, requires_grad=True)
        self.optimizer = RiemannianAdam(
            [self.X], lr=self.params.get('lr', 0.03)
        )

    def step(self, dt: float = 0.01, n_steps: int = 1) -> Tuple[Tensor, float]:
        """Run optimization steps minimizing spring + charge energy.

        Returns:
            positions: (N, dim) current positions (detached).
            energy: total energy after the last step.
        """
        spring_k = self.params['spring_k']
        target_L = self.params['target_L']
        charge_c = self.params['charge_c']
        eps = 1e-6

        total_energy = 0.0

        for _ in range(n_steps):
            self.optimizer.zero_grad()

            # Spring energy
            e_spring = torch.tensor(0.0, dtype=self.dtype, device=self.device)
            if self.edges:
                edge_tensor = torch.tensor(
                    self.edges, dtype=torch.long, device=self.device
                )
                i_idx, j_idx = edge_tensor[:, 0], edge_tensor[:, 1]
                d = self.manifold.dist(
                    self.X[i_idx], self.X[j_idx]
                ).squeeze(-1).clamp_min(eps)
                e_spring = 0.5 * spring_k * ((d - target_L) ** 2).sum()

            # Charge repulsion energy (pairwise)
            d_pairwise = self.manifold.dist(
                self.X.unsqueeze(1),
                self.X.unsqueeze(0)
            ).squeeze(-1).clamp_min(eps)

            iu, ju = torch.triu_indices(
                self.N, self.N, offset=1, device=self.device
            )
            e_charge = (charge_c ** 2 / d_pairwise[iu, ju]).sum()

            e_total = e_spring + e_charge
            e_total.backward()
            self.optimizer.step()

            total_energy = e_total.item()

        return self.X.detach(), total_energy

    def get_positions(self) -> Tensor:
        return self.X.detach()

    def update_params(self, params: Dict):
        self.params.update(params)
        if 'lr' in params:
            for pg in self.optimizer.param_groups:
                pg['lr'] = params['lr']
        if 'curvature' in params:
            self.c = params['curvature']
            self.manifold = PoincareBall(c=self.c)

    def reset(self, initial_positions: Optional[Tensor] = None):
        self._init_state(initial_positions)


# ---------------------------------------------------------------------------
# Geodesic Arc Computation
# ---------------------------------------------------------------------------

def compute_geodesic_arc(
    p1: Tensor,
    p2: Tensor,
    n_samples: int = 20,
) -> Tensor:
    """Compute geodesic arc between two points on the Poincare disk.

    Geodesics in the Poincare disk are circular arcs orthogonal to the
    unit circle boundary.

    Args:
        p1: (2,) first point in the disk.
        p2: (2,) second point in the disk.
        n_samples: number of sample points along the arc.
    Returns:
        arc: (n_samples, 2) points along the geodesic arc.
    """
    cross = p1[0] * p2[1] - p1[1] * p2[0]
    ts = torch.linspace(0.0, 1.0, n_samples, dtype=p1.dtype, device=p1.device)

    if abs(cross) < 1e-12:
        # Collinear with origin: geodesic is a straight line
        arc = (1 - ts).unsqueeze(1) * p1.unsqueeze(0) + ts.unsqueeze(1) * p2.unsqueeze(0)
    else:
        v = p2 - p1
        m = 0.5 * (p1 + p2)
        n = torch.tensor([-v[1], v[0]], dtype=p1.dtype, device=p1.device)
        n_norm = torch.norm(n)

        if n_norm < 1e-12:
            arc = (1 - ts).unsqueeze(1) * p1.unsqueeze(0) + ts.unsqueeze(1) * p2.unsqueeze(0)
        else:
            n = n / n_norm
            mn = torch.dot(m, n)
            rhs = 1.0 - (torch.dot(m, m) - 0.25 * torch.dot(v, v))
            denom = 2.0 * mn

            if abs(denom) < 1e-12:
                arc = (1 - ts).unsqueeze(1) * p1.unsqueeze(0) + ts.unsqueeze(1) * p2.unsqueeze(0)
            else:
                t_param = rhs / denom
                center = m + t_param * n
                r = torch.norm(center - p1)

                ang_p = torch.atan2(p1[1] - center[1], p1[0] - center[0])
                ang_q = torch.atan2(p2[1] - center[1], p2[0] - center[0])
                d_ang = ((ang_q - ang_p + math.pi) % (2 * math.pi)) - math.pi

                angs = ang_p + d_ang * ts
                arc = torch.stack([
                    center[0] + r * torch.cos(angs),
                    center[1] + r * torch.sin(angs),
                ], dim=1)

    return arc
