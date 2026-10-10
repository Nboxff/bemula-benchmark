"""Shared helpers for loading the YAML pipeline config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file (see configs/default.yaml)."""
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)
