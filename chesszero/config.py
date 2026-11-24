"""Load configs/base.yaml merged with one condition file.

Read the config once at the start of a script, then pass plain values into
functions and classes. Library code (model, MCTS, loss) shouldn't open YAML itself.

Example:
    cfg = load_config("configs/condition_25.yaml")
    require_complete(cfg)  # refuse to run with settings still set to null
    model = ChessNet.from_config(cfg)
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def _merge(base: dict, override: dict) -> dict:
    """Return base with override's values on top. Nested dicts merge key by key."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(condition_path=None) -> dict:
    """Load base.yaml, plus a condition file's overrides if one is given."""
    with open(BASE_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if condition_path is not None:
        with open(condition_path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def missing_settings(cfg: dict, prefix: str = "") -> list:
    """Return the dotted names of every setting that is still null."""
    missing = []
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if value is None:
            missing.append(name)
        elif isinstance(value, dict):
            missing.extend(missing_settings(value, name + "."))
    return missing


def require_complete(cfg: dict, sections=None) -> None:
    """Raise if any setting (optionally only within the given top-level sections) is null."""
    scope = cfg if sections is None else {s: cfg[s] for s in sections}
    missing = missing_settings(scope)
    if missing:
        raise ValueError("These settings are still null in the config: " + ", ".join(missing))
