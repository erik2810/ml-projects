"""Framework-native GNN reference implementation.

Wraps the existing :mod:`backend.core.gnn` layers and models behind the
``core.BaseModel`` / ``core.BaseDataset`` interfaces so they can be driven by
:class:`core.Trainer`, declared in YAML configs, and looked up by name.

Registered components:

- ``gcn_node``    - GCN-stack node classifier (BaseModel).
- ``gat_node``    - GAT-stack node classifier (BaseModel).
- ``gcn_graph``   - GCN-stack graph classifier (BaseModel).
- ``karate_club`` - Zachary's karate club full-batch dataset (BaseDataset).
"""

from __future__ import annotations

from algorithms.gnn.dataset import KarateClubDataset
from algorithms.gnn.model import (
    GATNodeClassifier,
    GCNGraphClassifier,
    GCNNodeClassifier,
)

__all__ = [
    "GCNNodeClassifier",
    "GATNodeClassifier",
    "GCNGraphClassifier",
    "KarateClubDataset",
]
