"""
YAML configuration loading, saving, and merging utilities.
"""

import copy
import os
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return it as a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """Persist a config dict to YAML (creates parent dirs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_config_with_overrides(
    base_cfg: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge a flat (possibly dotted) overrides dict into a nested base config.

    Example: overrides = {"training.epochs": 5} → cfg["training"]["epochs"] = 5.
    """
    cfg = copy.deepcopy(base_cfg)
    for key_path, value in overrides.items():
        parts = key_path.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg
