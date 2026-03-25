"""Riemannian manifolds for hyperbolic geometry: Poincare ball and Lorentz model."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device():
    """Detect best available device. Uses CPU for consistency with the rest of ml-projects."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Abstract Manifold
# ---------------------------------------------------------------------------

class Manifold(ABC):
    """Abstract base class for Riemannian manifolds."""

    def __init__(self, name: str = "Manifold"):
        self.name = name
        self._dim: Optional[int] = None

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    def __repr__(self) -> str:
        if self._dim is not None:
            return f"{self.name}(dim={self._dim})"
        return f"{self.name}()"

    @abstractmethod
    def projx(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        pass

    def norm(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.inner(x, u, u).clamp(min=1e-10))

    def transp(self, x: torch.Tensor, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.proju(y, u)

    def geodesic(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v = self.logmap(x, y)
        if t.dim() == 0:
            t = t.unsqueeze(-1)
        return self.expmap(x, t * v)


# ---------------------------------------------------------------------------
# Poincare Ball
# ---------------------------------------------------------------------------

class PoincareBall(Manifold):
    """Poincare ball model of hyperbolic space with curvature K = -c."""

    def __init__(self, c: float = 1.0):
        super().__init__(name=f"PoincareBall(c={c})")
        self.c = c
        self._sqrt_c = math.sqrt(c)

    def projx(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        max_norm = 1.0 - eps
        norm = torch.norm(x, dim=-1, keepdim=True)
        factor = torch.where(
            norm > max_norm,
            max_norm / norm,
            torch.ones_like(norm)
        )
        return x * factor

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u

    def _conformal_factor(self, x: torch.Tensor) -> torch.Tensor:
        norm_sq = (x * x).sum(dim=-1, keepdim=True)
        return 2.0 / (1.0 - self.c * norm_sq).clamp(min=1e-10)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        neg_x = -x
        mob = self.mobius_add(neg_x, y)
        mob_norm = torch.norm(mob, dim=-1, keepdim=True).clamp(min=1e-10)
        arg = (self._sqrt_c * mob_norm).clamp(max=1.0 - 1e-7)
        artanh = 0.5 * torch.log((1 + arg) / (1 - arg).clamp(min=1e-10))
        return (2.0 / self._sqrt_c) * artanh

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_sq = (x * x).sum(dim=-1, keepdim=True)
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        c = self.c
        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_sq * y_sq
        return num / denom.clamp(min=1e-10)

    def mobius_matvec(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Mx = x @ M.T
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-10)
        Mx_norm = torch.norm(Mx, dim=-1, keepdim=True).clamp(min=1e-10)
        arg = (self._sqrt_c * x_norm).clamp(max=1.0 - 1e-7)
        artanh = 0.5 * torch.log((1 + arg) / (1 - arg).clamp(min=1e-10))
        scale = torch.tanh(Mx_norm / x_norm * artanh) / (self._sqrt_c * Mx_norm)
        result = scale * Mx
        return self.projx(result)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        u_norm = torch.norm(u, dim=-1, keepdim=True).clamp(min=1e-10)
        lambda_x = self._conformal_factor(x)
        coef = torch.tanh(self._sqrt_c * lambda_x * u_norm / 2) / (self._sqrt_c * u_norm)
        second = coef * u
        small_u = u_norm < 1e-10
        second = torch.where(small_u, torch.zeros_like(u), second)
        result = self.mobius_add(x, second)
        return self.projx(result)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        neg_x = -x
        diff = self.mobius_add(neg_x, y)
        diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-10)
        lambda_x = self._conformal_factor(x)
        arg = (self._sqrt_c * diff_norm).clamp(max=1.0 - 1e-7)
        artanh = 0.5 * torch.log((1 + arg) / (1 - arg).clamp(min=1e-10))
        coef = (2.0 / (self._sqrt_c * lambda_x)) * artanh / diff_norm
        small_diff = diff_norm < 1e-10
        coef = torch.where(small_diff, torch.ones_like(coef), coef)
        return coef * diff

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        lambda_x = self._conformal_factor(x)
        return lambda_x.squeeze(-1) ** 2 * (u * v).sum(dim=-1)

    def norm(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        lambda_x = self._conformal_factor(x)
        return lambda_x.squeeze(-1) * torch.norm(u, dim=-1)

    def transp(self, x: torch.Tensor, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        lambda_x = self._conformal_factor(x)
        lambda_y = self._conformal_factor(y)
        return u * lambda_x / lambda_y

    def geodesic(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v = self.logmap(x, y)
        if t.dim() == 0:
            t = t.unsqueeze(-1)
        return self.expmap(x, t * v)

    def to_klein(self, x: torch.Tensor) -> torch.Tensor:
        x_sq = (x * x).sum(dim=-1, keepdim=True)
        return 2 * x / (1 + self.c * x_sq)

    def to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        x_sq = (x * x).sum(dim=-1, keepdim=True)
        scale = 2 / (1 - self.c * x_sq)
        t = (1 + self.c * x_sq) * scale / (2 * self._sqrt_c)
        spatial = x * scale / self._sqrt_c
        return torch.cat([t, spatial], dim=-1)

    def proj(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        return self.projx(x, eps=eps)

    def proj_tan(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.proju(x, u)

    def expmap0(self, u: torch.Tensor) -> torch.Tensor:
        """Exponential map from the origin."""
        origin = torch.zeros_like(u)
        return self.expmap(origin, u)

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map to the origin."""
        origin = torch.zeros_like(y)
        return self.logmap(origin, y)


# ---------------------------------------------------------------------------
# Lorentz (Hyperboloid) Model
# ---------------------------------------------------------------------------

class Lorentz(Manifold):
    """Lorentz (hyperboloid) model of hyperbolic space with curvature K = -k."""

    def __init__(self, k: float = 1.0):
        super().__init__(name=f"Lorentz(k={k})")
        self.k = k
        self._sqrt_k = math.sqrt(k)

    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        time_part = -x[..., 0] * y[..., 0]
        space_part = (x[..., 1:] * y[..., 1:]).sum(dim=-1)
        return time_part + space_part

    def projx(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        spatial_sq = (x[..., 1:] ** 2).sum(dim=-1, keepdim=True)
        time_sq = 1.0 / self.k + spatial_sq
        time = torch.sqrt(time_sq.clamp(min=eps))
        return torch.cat([time, x[..., 1:]], dim=-1)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inner = self.minkowski_inner(x, u)
        coef = -self.k * inner
        return u + coef.unsqueeze(-1) * x

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inner = self.minkowski_inner(x, y)
        arg = (-self.k * inner).clamp(min=1.0 + 1e-7)
        arcosh = torch.log(arg + torch.sqrt(arg ** 2 - 1))
        return arcosh / self._sqrt_k

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        u = self.proju(x, u)
        u_norm_sq = self.minkowski_inner(u, u)
        u_norm = torch.sqrt(u_norm_sq.clamp(min=1e-10)).unsqueeze(-1)
        small_u = u_norm < 1e-10
        sqrt_k_norm = self._sqrt_k * u_norm
        cosh_term = torch.cosh(sqrt_k_norm)
        sinh_term = torch.sinh(sqrt_k_norm) / sqrt_k_norm.clamp(min=1e-10)
        result = cosh_term * x + sinh_term * u
        result = torch.where(small_u, self.projx(x + u), result)
        return self.projx(result)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inner = self.minkowski_inner(x, y)
        dist = self.dist(x, y)
        v = y + self.k * inner.unsqueeze(-1) * x
        v_norm_sq = self.minkowski_inner(v, v)
        v_norm = torch.sqrt(v_norm_sq.clamp(min=1e-10)).unsqueeze(-1)
        small_dist = dist.unsqueeze(-1) < 1e-10
        result = dist.unsqueeze(-1) * v / v_norm
        result = torch.where(small_dist, self.proju(x, y - x), result)
        return result

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.minkowski_inner(u, v)

    def norm(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inner = self.minkowski_inner(u, u)
        return torch.sqrt(inner.clamp(min=1e-10))

    def transp(self, x: torch.Tensor, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        v = self.logmap(x, y)
        dist = self.dist(x, y)
        small_dist = dist < 1e-10
        v_norm = torch.sqrt(self.minkowski_inner(v, v).clamp(min=1e-10))
        v_prime = self.logmap(y, x)
        inner_vu = self.minkowski_inner(v, u)
        coef = inner_vu / (v_norm ** 2).clamp(min=1e-10)
        transported = u - coef.unsqueeze(-1) * (v + v_prime)
        transported = torch.where(
            small_dist.unsqueeze(-1),
            self.proju(y, u),
            transported
        )
        return self.proju(y, transported)

    def to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 1:] / (x[..., 0:1] + 1.0 / self._sqrt_k)

    def from_poincare(self, p: torch.Tensor) -> torch.Tensor:
        p_sq = (p ** 2).sum(dim=-1, keepdim=True)
        denom = (1 - self.k * p_sq).clamp(min=1e-10)
        time = (1 + self.k * p_sq) / (self._sqrt_k * denom)
        spatial = 2 * p / denom
        return torch.cat([time, spatial], dim=-1)

    def centroid(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute hyperbolic centroid via Frechet mean."""
        n = x.shape[0]
        if weights is None:
            weights = torch.ones(n, device=x.device, dtype=x.dtype) / n
        else:
            weights = weights / weights.sum()

        # Initialize at weighted Euclidean mean projected onto hyperboloid
        mean = (x * weights.unsqueeze(-1)).sum(dim=0)
        mean = self.projx(mean)

        for _ in range(100):
            tangent_sum = torch.zeros_like(mean)
            for i in range(n):
                log_vec = self.logmap(mean, x[i])
                tangent_sum = tangent_sum + weights[i] * log_vec
            step_norm = torch.norm(tangent_sum)
            if step_norm < 1e-6:
                break
            mean = self.expmap(mean, tangent_sum)

        return mean

    def origin(self, dim: int, dtype: torch.dtype = torch.float32,
               device: torch.device = None) -> torch.Tensor:
        o = torch.zeros(dim, dtype=dtype, device=device)
        o[0] = 1.0 / self._sqrt_k
        return o


# ---------------------------------------------------------------------------
# ManifoldParameter
# ---------------------------------------------------------------------------

class ManifoldParameter(nn.Parameter):
    """nn.Parameter constrained to a Riemannian manifold."""

    def __new__(cls, data: Optional[torch.Tensor] = None,
                manifold: Optional[Manifold] = None,
                requires_grad: bool = True) -> 'ManifoldParameter':
        if data is None:
            data = torch.empty(0)
        instance = super().__new__(cls, data, requires_grad)
        return instance

    def __init__(self, data: Optional[torch.Tensor] = None,
                 manifold: Optional[Manifold] = None,
                 requires_grad: bool = True):
        self.manifold = manifold

    def __repr__(self) -> str:
        manifold_str = self.manifold.name if self.manifold else "None"
        return f"ManifoldParameter({self.data.shape}, manifold={manifold_str})"


# ---------------------------------------------------------------------------
# Riemannian Adam Optimizer
# ---------------------------------------------------------------------------

class RiemannianAdam(torch.optim.Optimizer):
    """Adam optimizer respecting Riemannian manifold geometry."""

    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0, amsgrad: bool = False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                manifold = getattr(p, 'manifold', None)
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1

                if manifold is not None:
                    rgrad = manifold.proju(p.data, grad)
                    exp_avg.mul_(beta1).add_(rgrad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(rgrad, rgrad, value=1 - beta2)

                    if amsgrad:
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = max_exp_avg_sq.sqrt().add_(eps)
                    else:
                        denom = exp_avg_sq.sqrt().add_(eps)

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                    direction = exp_avg / denom
                    p.data = manifold.expmap(p.data, -step_size * direction)
                    p.data = manifold.projx(p.data)
                else:
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    if amsgrad:
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = max_exp_avg_sq.sqrt().add_(eps)
                    else:
                        denom = exp_avg_sq.sqrt().add_(eps)

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
