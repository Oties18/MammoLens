"""Config loading helpers.

Keeps a single, typed entry point for reading configs/config.yaml so every
module (data, model, train, evaluate, gradcam) sees the same values.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import yaml


def _to_namespace(obj):
    """Recursively turn nested dicts into attribute-accessible namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str = "configs/config.yaml") -> SimpleNamespace:
    """Load YAML config and return it as a nested namespace.

    Access values as cfg.train.batch_size, cfg.paths.data_dir, etc.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


def ensure_dirs(cfg: SimpleNamespace) -> None:
    """Create checkpoint/log directories on Drive if they don't exist yet."""
    for d in (cfg.paths.checkpoint_dir, cfg.paths.log_dir):
        os.makedirs(d, exist_ok=True)
