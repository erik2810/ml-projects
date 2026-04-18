"""Framework-native hyperbolic GNN.

Wraps :class:`backend.core.hyperbolic.models.HyperbolicGNN` and
:class:`HyperbolicEmbedding` under the ``core.BaseModel`` interface with
Riemannian + Euclidean optimizer support.

Registered components:

- ``hyperbolic_gnn``         - Poincaré-ball node classifier (BaseModel).
- ``hyperbolic_embedding``   - Learnable Poincaré embeddings for link prediction.
- ``tree_graph``             - Synthetic rooted tree dataset (BaseDataset).
"""

from __future__ import annotations

from algorithms.hyperbolic.dataset import TreeGraphDataset
from algorithms.hyperbolic.model import HyperbolicGNNModel, HyperbolicEmbeddingModel

__all__ = ["HyperbolicGNNModel", "HyperbolicEmbeddingModel", "TreeGraphDataset"]
