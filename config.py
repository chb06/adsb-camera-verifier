from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open('r') as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge upd into base."""
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(default_path: str | Path, override_path: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg = load_yaml(default_path)
    if override_path:
        cfg = deep_update(cfg, load_yaml(override_path))
    return cfg
