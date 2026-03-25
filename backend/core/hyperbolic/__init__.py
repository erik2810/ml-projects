"""Hyperbolic GNN core module -- geometry-aware graph neural networks on Riemannian manifolds."""

from .manifolds import (
    Manifold,
    PoincareBall,
    Lorentz,
    ManifoldParameter,
    RiemannianAdam,
    get_device,
)

from .layers import (
    HyperbolicMessagePassing,
    HyperbolicGCNLayer,
    HyperbolicGATLayer,
)

from .models import (
    HyperbolicGNN,
    HyperbolicEmbedding,
)

from .simulation import (
    HyperbolicForces,
    HyperbolicSimulation,
    compute_geodesic_arc,
)

from .train import (
    HyperbolicTrainConfig,
    train_hyperbolic_gnn,
    train_hyperbolic_embedding,
    compare_embeddings,
)

__all__ = [
    # Manifolds
    'Manifold',
    'PoincareBall',
    'Lorentz',
    'ManifoldParameter',
    'RiemannianAdam',
    'get_device',
    # Layers
    'HyperbolicMessagePassing',
    'HyperbolicGCNLayer',
    'HyperbolicGATLayer',
    # Models
    'HyperbolicGNN',
    'HyperbolicEmbedding',
    # Simulation
    'HyperbolicForces',
    'HyperbolicSimulation',
    'compute_geodesic_arc',
    # Training
    'HyperbolicTrainConfig',
    'train_hyperbolic_gnn',
    'train_hyperbolic_embedding',
    'compare_embeddings',
]
