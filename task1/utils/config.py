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
    if cfg is None:
        return {}
    return cfg


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """Persist a config dict to a YAML file (creates parent dirs if needed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base.

    Values in override take precedence; nested dicts are merged recursively
    rather than replaced wholesale.
    """
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
    Merge a flat overrides dict into a nested base config.

    Overrides keys may use dot notation for nested paths, e.g.:
        "training.backbone_lr" → cfg["training"]["backbone_lr"]

    Args:
        base_cfg: The base configuration dict (loaded from YAML).
        overrides: Flat dict of key→value overrides.

    Returns:
        New merged config dict (base_cfg is not modified).
    """
    cfg = copy.deepcopy(base_cfg)
    for key_path, value in overrides.items():
        parts = key_path.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg
