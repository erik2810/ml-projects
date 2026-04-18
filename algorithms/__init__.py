"""Framework-native algorithm implementations.

Each submodule under ``algorithms/`` exposes a :class:`core.BaseModel` and a
:class:`core.BaseDataset` (plus optional callbacks) wired up through the
component registry. Importing a submodule is sufficient to register its
components — downstream code can then look them up by string name via
:func:`core.registry.get`.

Algorithms currently available:

- :mod:`algorithms.gnn` - GCN/GAT node & graph classification on Karate Club.

The legacy numeric implementations under ``backend/core/*`` remain importable
and unchanged; ``algorithms/*`` wraps them (or re-implements on top of
``core/``) so experiments can be specified declaratively.
"""

from __future__ import annotations

from algorithms import gnn  # noqa: F401 — import side effect registers components

__all__ = ["gnn"]
