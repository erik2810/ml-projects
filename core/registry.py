"""String-keyed registry for pluggable components.

A minimal inversion-of-control container so experiments can refer to models,
datasets, and trainers by name in YAML configs without hard imports.

Example::

    @register("gcn", kind="model")
    class GCNNodeClassifier(BaseModel):
        ...

    cls = get("gcn", kind="model")
    model = cls(**config.model.params)
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

_REGISTRIES: dict[str, dict[str, type]] = {
    "model": {},
    "dataset": {},
    "trainer": {},
    "callback": {},
}


def register(name: str, *, kind: str = "model") -> Callable[[type[T]], type[T]]:
    """Decorator that registers a class under ``name`` for later retrieval."""
    if kind not in _REGISTRIES:
        _REGISTRIES[kind] = {}

    def decorator(cls: type[T]) -> type[T]:
        if name in _REGISTRIES[kind]:
            existing = _REGISTRIES[kind][name]
            raise ValueError(f"{kind} '{name}' is already registered to {existing!r}")
        _REGISTRIES[kind][name] = cls
        return cls

    return decorator


def get(name: str, *, kind: str = "model") -> type:
    """Retrieve a registered class by name."""
    if kind not in _REGISTRIES or name not in _REGISTRIES[kind]:
        available = sorted(_REGISTRIES.get(kind, {}).keys())
        raise KeyError(
            f"{kind} '{name}' is not registered. Available {kind}s: {available or '(none)'}"
        )
    return _REGISTRIES[kind][name]


def list_registered(kind: str | None = None) -> dict[str, list[str]]:
    """Return the contents of one or all registries as sorted name lists."""
    if kind is not None:
        return {kind: sorted(_REGISTRIES.get(kind, {}).keys())}
    return {k: sorted(v.keys()) for k, v in _REGISTRIES.items()}
