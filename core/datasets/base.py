"""Dataset protocol used by the Trainer.

Unlike image/text ML, graph workloads in this project are dominated by
full-batch semi-supervised settings (Karate Club, small meshes). A dataset
therefore yields *batches* that may be anything — a single graph tuple,
a list of graphs, or a proper ``DataLoader`` — and the Trainer stays agnostic.

Implementers have two options:

1. Subclass :class:`BaseDataset` and return iterables from
   :meth:`train_batches` / :meth:`val_batches` / :meth:`test_batches`.
2. Pass any object that matches the protocol (duck-typed). Anything with
   ``train_batches()`` callable is accepted.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable


class BaseDataset:
    """Default dataset base. Subclasses override batch generators.

    The Trainer calls these lazily each epoch, so generators are allowed
    (and preferred for large data). For full-batch training, a list of a
    single element is the canonical pattern.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = dict(config) if config else {}

    def train_batches(self) -> Iterable[Any]:
        raise NotImplementedError

    def val_batches(self) -> Iterable[Any] | None:
        return None

    def test_batches(self) -> Iterable[Any] | None:
        return None

    def get_config(self) -> dict:
        return dict(self._config)


@runtime_checkable
class DatasetProtocol(Protocol):
    """Structural type: any object with a ``train_batches()`` method works."""

    def train_batches(self) -> Iterable[Any]: ...
