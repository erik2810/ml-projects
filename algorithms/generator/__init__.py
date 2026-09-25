"""Framework-native conditional graph generator.

Adapts :class:`backend.core.generator.model.ConditionalGraphGenerator` to the
``core.BaseModel`` interface and exposes its training set via
:class:`ConditionalGraphsDataset`.

Registered components:

- ``cond_graph_vae``     - CVAE that generates graphs conditioned on [nodes, density, clustering].
- ``random_graph_bank``  - ER + BA mixture with condition vectors (BaseDataset).
"""

from __future__ import annotations

from algorithms.generator.dataset import ConditionalGraphsDataset
from algorithms.generator.model import ConditionalGenerator

__all__ = ["ConditionalGenerator", "ConditionalGraphsDataset"]
