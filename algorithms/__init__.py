"""Framework-native algorithm implementations.

Each submodule under ``algorithms/`` exposes a :class:`core.BaseModel` and a
:class:`core.BaseDataset` (plus optional callbacks) wired up through the
component registry. Importing a submodule is sufficient to register its
components — downstream code can then look them up by string name via
:func:`core.registry.get`.

Algorithms currently available:

- :mod:`algorithms.gnn`         - GCN/GAT node & graph classification.
- :mod:`algorithms.graphvae`    - Graph VAE and discrete graph diffusion.
- :mod:`algorithms.generator`   - Conditional graph VAE (CVAE).
- :mod:`algorithms.hyperbolic`  - Poincaré-ball GNN and embeddings.
- :mod:`algorithms.physics_gnn` - Physics-informed GNN on 3D meshes.

The legacy numeric implementations under ``backend/core/*`` remain importable
and unchanged; ``algorithms/*`` wraps them so experiments can be specified
declaratively via YAML configs.
"""

from __future__ import annotations

from algorithms import (
    generator,  # noqa: F401
    gnn,  # noqa: F401
    graphvae,  # noqa: F401
    hyperbolic,  # noqa: F401
    physics_gnn,  # noqa: F401
)

__all__ = ["generator", "gnn", "graphvae", "hyperbolic", "physics_gnn"]
