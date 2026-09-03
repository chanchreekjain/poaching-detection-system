"""Configuration loading.

A thin wrapper over the YAML file so the rest of the code can do
``cfg.motion.hold_frames`` instead of ``cfg["motion"]["hold_frames"]``,
without pulling in a dependency like pydantic or omegaconf.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


class Namespace(dict):
    """dict that also supports attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"no config key {item!r} (available: {sorted(self)})"
            ) from exc
        return Namespace(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _resolve_paths(node: Any) -> Any:
    """Make relative paths in the config relative to the project root."""
    if isinstance(node, dict):
        return {k: _resolve_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_paths(v) for v in node]
    return node


def load_config(path: str | Path | None = None) -> Namespace:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Namespace(_resolve_paths(copy.deepcopy(raw)))


def resolve(path_like: str | Path) -> Path:
    """Resolve a config path relative to the project root if it is relative."""
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
