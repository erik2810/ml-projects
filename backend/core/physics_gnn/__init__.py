"""
Physics-informed graph neural networks grounded in differential geometry.

Bridges discrete differential geometry (DDG) and graph representation learning
by replacing heuristic message-passing weights with geometry-derived operators.
The core insight: the cotangent Laplacian — the standard discretisation of the
Laplace–Beltrami operator on surfaces — is exactly the weighted adjacency used
in spectral graph convolution.  This package exploits that equivalence to build
GNNs whose inductive biases come from physics rather than architecture search.

Modules
-------
operators
    Cotangent Laplacian, geometric edge weights, discrete curvatures,
    heat kernels, multiscale diffusion filters, geodesic distances.
layers
    CotangentConv, DiffusionConv, ReactionDiffusionLayer,
    ManifoldMessagePassing, CurvatureAttention, GeometricEdgeEncoder.
energy
    Dirichlet energy, total variation, Willmore energy, elastic energy,
    and the combined PhysicsRegulariser with learnable task weighting.
models
    PhysicsInformedGNN for node/graph prediction tasks.
    PhysicsInformedGraphGenerator for spatial graph generation via
    reaction-diffusion morphogenesis.
train
    Training loops, ablation study runner, experiment logging.
"""

# Geometric operators
from .operators import (
    cotangent_laplacian,
    geometric_edge_weights,
    weighted_laplacian,
    symmetric_normalised_laplacian,
    discrete_curvatures,
    heat_kernel,
    multiscale_diffusion_filters,
    heat_method_distances,
)

# Neural network layers
from .layers import (
    CotangentConv,
    DiffusionConv,
    ReactionDiffusionLayer,
    ManifoldMessagePassing,
    CurvatureAttention,
    GeometricEdgeEncoder,
)

# Energy functionals and regularisation
from .energy import (
    dirichlet_energy,
    dirichlet_energy_from_positions,
    total_variation,
    willmore_energy,
    elastic_energy,
    PhysicsRegulariser,
)

# Model architectures
from .models import (
    PhysicsInformedGNN,
    PhysicsInformedGraphGenerator,
)

# Training utilities
from .train import (
    TrainConfig,
    AblationResult,
    train_node_model,
    train_generator,
    run_ablation,
    print_ablation_table,
)

__all__ = [
    # operators
    'cotangent_laplacian',
    'geometric_edge_weights',
    'weighted_laplacian',
    'symmetric_normalised_laplacian',
    'discrete_curvatures',
    'heat_kernel',
    'multiscale_diffusion_filters',
    'heat_method_distances',
    # layers
    'CotangentConv',
    'DiffusionConv',
    'ReactionDiffusionLayer',
    'ManifoldMessagePassing',
    'CurvatureAttention',
    'GeometricEdgeEncoder',
    # energy
    'dirichlet_energy',
    'dirichlet_energy_from_positions',
    'total_variation',
    'willmore_energy',
    'elastic_energy',
    'PhysicsRegulariser',
    # models
    'PhysicsInformedGNN',
    'PhysicsInformedGraphGenerator',
    # train
    'TrainConfig',
    'AblationResult',
    'train_node_model',
    'train_generator',
    'run_ablation',
    'print_ablation_table',
]
