"""Framework-native physics-informed GNN.

Adapts :class:`backend.core.physics_gnn.models.PhysicsInformedGNN` to the
``core.BaseModel`` interface.

Registered components:

- ``physics_gnn_node``   - Physics-informed node classifier (BaseModel).
- ``spring_mesh``        - Synthetic spring-mesh dataset (BaseDataset).
- ``egnn_denoise``       - E(n)-equivariant coordinate denoiser (BaseModel).
- ``noisy_mesh``         - Noised-mesh dataset for the denoiser (BaseDataset).
"""

from __future__ import annotations

from algorithms.physics_gnn.dataset import SpringMeshDataset
from algorithms.physics_gnn.egnn import EGNNDenoiseModel, NoisyMeshDataset
from algorithms.physics_gnn.model import PhysicsGNNNodeModel

__all__ = [
    "EGNNDenoiseModel",
    "NoisyMeshDataset",
    "PhysicsGNNNodeModel",
    "SpringMeshDataset",
]
