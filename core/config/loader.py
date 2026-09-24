"""YAML / JSON config loader.

YAML is preferred (via optional ``PyYAML``); JSON is always supported via the
standard library as a fallback. Both formats are interchangeable — the loader
dispatches on file extension.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config.schema import ExperimentConfig

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover - PyYAML is optional
    _yaml = None  # type: ignore[assignment]


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from ``path`` (YAML or JSON)."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    data = _parse(raw, path.suffix)
    return ExperimentConfig.model_validate(data)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    """Serialize ``config`` to ``path`` (format chosen by extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump()
    if path.suffix in (".yaml", ".yml"):
        _require_yaml()
        path.write_text(_yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse(text: str, suffix: str) -> dict[str, Any]:
    if suffix in (".yaml", ".yml"):
        _require_yaml()
        return _yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text)
    # Unknown extension. YAML reads most JSON, but PyYAML rejects some valid
    # JSON (tab indentation), so fall back to json. If that fails too, the YAML
    # error is the one worth showing.
    if _yaml is None:
        return json.loads(text)
    try:
        return _yaml.safe_load(text) or {}
    except _yaml.YAMLError as yaml_err:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise yaml_err from None


def _require_yaml() -> None:
    if _yaml is None:
        raise ImportError(
            "PyYAML is required to read/write YAML configs. "
            "Install it with `pip install pyyaml` or use a .json file instead."
        )
