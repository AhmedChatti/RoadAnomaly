"""Load and validate the project configuration."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.json"


def project_root() -> Path:
    return ROOT


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path from configs.default.json paths section against the project root."""
    relative = config["paths"][key]
    path = Path(relative)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


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
