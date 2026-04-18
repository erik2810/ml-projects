"""Framework-native physics-informed GNN.

Adapts :class:`backend.core.physics_gnn.models.PhysicsInformedGNN` to the
``core.BaseModel`` interface.

Registered components:

- ``physics_gnn_node``   - Physics-informed node classifier (BaseModel).
- ``spring_mesh``        - Synthetic spring-mesh dataset (BaseDataset).
"""

from __future__ import annotations

from algorithms.physics_gnn.dataset import SpringMeshDataset
from algorithms.physics_gnn.model import PhysicsGNNNodeModel

__all__ = ["PhysicsGNNNodeModel", "SpringMeshDataset"]
