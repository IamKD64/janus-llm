"""YAML config loading with dotted-key CLI overrides (``--set train.epochs=1``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg
