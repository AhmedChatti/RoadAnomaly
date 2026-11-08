"""Load and validate the project configuration."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.json"

# Optional Colab / Drive overrides (infrastructure only; does not change scientific protocol).
_ENV_PATH_OVERRIDES = {
    "frames_dir": "ROADANOMALY_FRAMES_DIR",
    "checkpoints_dir": "ROADANOMALY_CHECKPOINTS_DIR",
    "experiments_metadata_dir": "ROADANOMALY_METADATA_DIR",
    "experiments_results_dir": "ROADANOMALY_RESULTS_DIR",
    "experiments_figures_dir": "ROADANOMALY_FIGURES_DIR",
}


def project_root() -> Path:
    env = os.environ.get("ROADANOMALY_ROOT")
    if env:
        return Path(env).resolve()
    return ROOT


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else (project_root() / "configs" / "default.json")
    if path is None and not config_path.is_file():
        config_path = DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path from configs.default.json paths section against the project root.

    Environment overrides (for Colab + Drive) take precedence when set:
    - ROADANOMALY_FRAMES_DIR
    - ROADANOMALY_CHECKPOINTS_DIR
    - ROADANOMALY_METADATA_DIR / RESULTS_DIR / FIGURES_DIR
    - ROADANOMALY_ARTIFACTS_ROOT: if set, relative path keys are resolved under this root
    """
    env_key = _ENV_PATH_OVERRIDES.get(key)
    if env_key:
        env_val = os.environ.get(env_key)
        if env_val:
            return Path(env_val).resolve()

    relative = config["paths"][key]
    path = Path(relative)
    if path.is_absolute():
        return path

    artifacts_root = os.environ.get("ROADANOMALY_ARTIFACTS_ROOT")
    if artifacts_root and key in {
        "checkpoints_dir",
        "experiments_metadata_dir",
        "experiments_results_dir",
        "experiments_figures_dir",
    }:
        return (Path(artifacts_root) / path).resolve()

    return (project_root() / path).resolve()


def preprocessing_size(config: dict[str, Any] | None = None) -> tuple[int, int]:
    cfg = config or load_config()
    height = int(cfg["preprocessing"]["input_height"])
    width = int(cfg["preprocessing"]["input_width"])
    return height, width


def assert_no_letterbox(config: dict[str, Any] | None = None) -> None:
    cfg = config or load_config()
    if cfg["preprocessing"].get("use_letterbox_padding", False):
        raise ValueError(
            "Protocol Phase 1 forbids letterbox padding for the training pipeline. "
            "Set preprocessing.use_letterbox_padding to false."
        )


def with_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    merged = deepcopy(config)
    for key, value in overrides.items():
        merged[key] = value
    return merged
