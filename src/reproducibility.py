"""Experiment reproducibility helpers for local and Colab runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import load_config, project_root, resolve_path
from dataset import assignment_sha256_from_manifest, load_split_manifest


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_sha(*, repo_dir: Path | None = None, fallback_env: str = "ROADANOMALY_GIT_SHA") -> str | None:
    """Return the current HEAD SHA, or an explicit override from the environment."""
    env_sha = os.environ.get(fallback_env)
    if env_sha:
        return env_sha.strip()
    root = repo_dir or project_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def software_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import numpy as np

        versions["numpy"] = np.__version__
    except Exception as exc:  # noqa: BLE001
        versions["numpy"] = f"unavailable: {exc}"
    try:
        import PIL

        versions["pillow"] = PIL.__version__
    except Exception as exc:  # noqa: BLE001
        versions["pillow"] = f"unavailable: {exc}"
    try:
        import tensorflow as tf
        from tensorflow import keras

        versions["tensorflow"] = tf.__version__
        versions["keras"] = keras.__version__
    except Exception as exc:  # noqa: BLE001
        versions["tensorflow"] = f"unavailable: {exc}"
        versions["keras"] = "unavailable"
    return versions


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "gpu_available": False,
        "gpu_devices": [],
        "gpu_model": None,
        "notes": None,
    }
    try:
        import tensorflow as tf

        devices = tf.config.list_physical_devices("GPU")
        info["gpu_devices"] = [d.name for d in devices]
        info["gpu_available"] = bool(devices)
        if devices:
            try:
                from tensorflow.python.client import device_lib

                details = device_lib.list_local_devices()
                gpu_names = [d.name + " | " + d.physical_device_desc for d in details if d.device_type == "GPU"]
                info["gpu_model"] = gpu_names[0] if gpu_names else devices[0].name
            except Exception:  # noqa: BLE001
                info["gpu_model"] = devices[0].name
        else:
            info["notes"] = (
                "No TensorFlow GPU device visible. "
                "On native Windows TF>=2.11 this is expected; use Colab GPU or WSL2."
            )
    except Exception as exc:  # noqa: BLE001
        info["notes"] = f"TensorFlow GPU probe failed: {exc}"
    return info


def dataset_fingerprint(frames_dir: Path | None = None) -> dict[str, Any]:
    """Lightweight fingerprint: counts + sorted frame_list hash (not full image hashing)."""
    cfg = load_config()
    frames = frames_dir or resolve_path(cfg, "frames_dir")
    frame_list = project_root() / "frame_list.json"
    names: list[str] = []
    if frame_list.is_file():
        names = list(json.loads(frame_list.read_text(encoding="utf-8")))
    webp = sorted(p.name for p in frames.glob("*.webp")) if frames.is_dir() else []
    labels = sorted(p.name for p in frames.glob("*.labels")) if frames.is_dir() else []
    return {
        "frames_dir": str(frames),
        "n_webp": len(webp),
        "n_label_dirs": len(labels),
        "frame_list_count": len(names),
        "frame_list_sha256": hashlib.sha256(
            json.dumps(names, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        if names
        else None,
        "webp_names_sha256": hashlib.sha256("\n".join(webp).encode("utf-8")).hexdigest() if webp else None,
    }


def split_lock_info() -> dict[str, Any]:
    cfg = load_config()
    manifest = load_split_manifest(cfg)
    return {
        "assignment_sha256": assignment_sha256_from_manifest(manifest),
        "expected_assignment_sha256": cfg["split"]["assignment_sha256"],
        "counts": {
            "train": int((manifest["split"] == "train").sum()),
            "val": int((manifest["split"] == "val").sum()),
            "test": int((manifest["split"] == "test").sum()),
        },
        "input_height": int(cfg["preprocessing"]["input_height"]),
        "input_width": int(cfg["preprocessing"]["input_width"]),
    }


def build_experiment_metadata_base(
    *,
    experiment_id: str,
    model_name: str,
    architecture: str,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Standard metadata skeleton for every Colab / local experiment."""
    cfg = load_config()
    split = split_lock_info()
    return {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha(),
        "dataset_fingerprint": dataset_fingerprint(),
        "split_manifest_sha256": split["assignment_sha256"],
        "split_lock_ok": split["assignment_sha256"] == split["expected_assignment_sha256"],
        "input_height": split["input_height"],
        "input_width": split["input_width"],
        "model_name": model_name,
        "architecture": architecture,
        "seed": int(cfg["training"]["seed"]),
        "hyperparameters": hyperparameters or {},
        "software_versions": software_versions(),
        "gpu": gpu_info(),
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "history": {},
        "validation_metrics": {},
        "threshold_selection": {
            "split": "val",
            "metric": "mean_dice",
            "value": None,
            "default_during_training": 0.5,
        },
        "artefact_locations": {},
        "status": "initialized",
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def verify_writable(directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return {"path": str(directory), "writable": True, "error": None}
    except OSError as exc:
        return {"path": str(directory), "writable": False, "error": str(exc)}


EXPECTED_PHASE3_CHECKPOINT_SHA256 = (
    "567bde326ee5b776bc34132b23012543b35f195ec8d60229d01196eb1a0fb234"
)
