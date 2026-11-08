"""Inference helpers for the RoadAnomaly Streamlit application (E2b only).

Scientific freeze (do not change without a new formal selection):
- Model: E2b U-Net + MobileNetV2 selective fine-tune
- Threshold: 0.55
- Input: 512×288 (W×H)
- Normalize: mobilenet_v2
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from checkpoint_io import load_keras_model_compat, load_weights_from_keras_zip
from metrics import threshold_probabilities
from models.unet_mobilenetv2 import build_unet_mobilenetv2
from preprocessing import normalize_image, resize_image_rgb, resize_mask_nearest

# Frozen selection — mirrors FINAL_MODEL_SELECTION_FREEZE.json
E2B_EXPERIMENT_ID = "E2b_unet_mobilenetv2_finetune_20261108_1721"
E2B_THRESHOLD = 0.55
E2B_INPUT_WIDTH = 512
E2B_INPUT_HEIGHT = 288
E2B_NORMALIZE = "mobilenet_v2"
E2B_CHECKPOINT_SHA256 = "280170dc64347b0b4cc8ce3af9be40dec4ac3733b27d1abeb620cc0c8e88e432"
DEFAULT_RELATIVE_CHECKPOINT = Path("checkpoints") / "phase4" / E2B_EXPERIMENT_ID / "best.keras"

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class InferenceResult:
    original_rgb: np.ndarray
    model_rgb: np.ndarray
    probability_map: np.ndarray
    binary_mask: np.ndarray
    probability_full: np.ndarray
    binary_full: np.ndarray
    overlay_rgb: np.ndarray
    anomaly_pixel_pct: float
    mean_anomaly_probability: float
    threshold: float
    input_width: int
    input_height: int


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_e2b_checkpoint(explicit: str | Path | None = None) -> Path:
    """Resolve checkpoint path without inventing an alternate model."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("ROADANOMALY_E2B_CHECKPOINT")
    if env:
        candidates.append(Path(env))
    root = project_root()
    candidates.append(root / DEFAULT_RELATIVE_CHECKPOINT)
    # Optional Colab / Drive mount path (never used as a different experiment).
    candidates.append(
        Path("/content/drive/MyDrive/RoadAnomaly")
        / "checkpoints"
        / "phase4"
        / E2B_EXPERIMENT_ID
        / "best.keras"
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Checkpoint E2b introuvable. Copiez best.keras depuis Google Drive vers :\n"
        f"  {root / DEFAULT_RELATIVE_CHECKPOINT}\n"
        "ou définissez ROADANOMALY_E2B_CHECKPOINT.\n"
        f"Chemins essayés :\n{searched}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_e2b_checkpoint_sha(path: Path, *, expected: str = E2B_CHECKPOINT_SHA256) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 du checkpoint E2b incorrect.\n  obtenu  : {actual}\n  attendu : {expected}"
        )
    return actual


def load_e2b_model(checkpoint: str | Path | None = None, *, verify_sha: bool = True) -> Any:
    """Load the frozen E2b weights once (caller should cache)."""
    path = resolve_e2b_checkpoint(checkpoint)
    if verify_sha:
        verify_e2b_checkpoint_sha(path)
    try:
        return load_keras_model_compat(path, compile=False)
    except Exception:
        model = build_unet_mobilenetv2(
            input_height=E2B_INPUT_HEIGHT,
            input_width=E2B_INPUT_WIDTH,
            weights=None,
            encoder_trainable=False,
        )
        load_weights_from_keras_zip(model, path)
        return model


def load_user_image(data: bytes | Path | str, filename: str | None = None) -> np.ndarray:
    """Load RGB uint8 from upload bytes or path; validate format."""
    if isinstance(data, (str, Path)):
        path = Path(data)
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"Format non supporté ({suffix}). Utilisez JPG, JPEG ou PNG.")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    else:
        name = (filename or "upload.bin").lower()
        suffix = Path(name).suffix.lower()
        if suffix and suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"Format non supporté ({suffix}). Utilisez JPG, JPEG ou PNG.")
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Image RGB attendue, reçu shape={array.shape}")
    if array.shape[0] < 8 or array.shape[1] < 8:
        raise ValueError("Image trop petite (minimum 8×8).")
    return array


def preprocess_for_e2b(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (batch float32 NHWC, resized uint8 model-size RGB)."""
    model_rgb = resize_image_rgb(image_rgb, E2B_INPUT_WIDTH, E2B_INPUT_HEIGHT)
    normalized = normalize_image(model_rgb, E2B_NORMALIZE).astype(np.float32)
    batch = np.expand_dims(normalized, axis=0)
    return batch, model_rgb


def make_overlay(
    image_rgb: np.ndarray,
    binary_mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (220, 50, 50),
    alpha: float = 0.45,
) -> np.ndarray:
    if image_rgb.shape[:2] != binary_mask.shape[:2]:
        raise ValueError("Overlay: image and mask spatial sizes must match")
    out = image_rgb.astype(np.float32).copy()
    mask = binary_mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_probability_bilinear(probs: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize HxW probability map to original size (display only; not used for threshold)."""
    # Threshold is applied at model resolution; this map is for visualization / download.
    pil = Image.fromarray((np.clip(probs, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    resized = pil.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def predict_e2b(
    model: Any,
    image_rgb: np.ndarray,
    *,
    threshold: float = E2B_THRESHOLD,
) -> InferenceResult:
    if abs(float(threshold) - E2B_THRESHOLD) > 1e-9:
        # Allow explicit override only for unit tests; production UI must pass freeze value.
        pass
    orig_h, orig_w = image_rgb.shape[:2]
    batch, model_rgb = preprocess_for_e2b(image_rgb)
    raw = model.predict(batch, verbose=0)
    probs = np.asarray(raw, dtype=np.float32)
    if probs.ndim == 4:
        probs = probs[0, ..., 0]
    elif probs.ndim == 3:
        probs = probs[..., 0]
    if probs.shape != (E2B_INPUT_HEIGHT, E2B_INPUT_WIDTH):
        raise RuntimeError(
            f"Sortie modèle inattendue {probs.shape}, attendu "
            f"({E2B_INPUT_HEIGHT}, {E2B_INPUT_WIDTH})"
        )
    if not np.isfinite(probs).all():
        raise RuntimeError("La carte de probabilités contient des valeurs non finies.")
    if probs.min() < -1e-3 or probs.max() > 1.0 + 1e-3:
        raise RuntimeError(
            f"Probabilités hors [0,1] (min={probs.min():.4f}, max={probs.max():.4f})."
        )
    probs = np.clip(probs, 0.0, 1.0)
    binary = threshold_probabilities(probs, threshold).astype(np.uint8)
    binary_full = resize_mask_nearest(binary, orig_w, orig_h)
    # resize_mask_nearest remaps labels; ensure {0,1}
    binary_full = (binary_full > 0).astype(np.uint8)
    prob_full = resize_probability_bilinear(probs, orig_w, orig_h)
    overlay = make_overlay(image_rgb, binary_full)
    anomaly_pct = float(binary_full.mean() * 100.0)
    mean_prob = float(probs.mean())
    return InferenceResult(
        original_rgb=image_rgb,
        model_rgb=model_rgb,
        probability_map=probs,
        binary_mask=binary,
        probability_full=prob_full,
        binary_full=binary_full,
        overlay_rgb=overlay,
        anomaly_pixel_pct=anomaly_pct,
        mean_anomaly_probability=mean_prob,
        threshold=float(threshold),
        input_width=E2B_INPUT_WIDTH,
        input_height=E2B_INPUT_HEIGHT,
    )


def encode_png_rgb(array: np.ndarray) -> bytes:
    image = Image.fromarray(array, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def encode_png_mask(mask: np.ndarray) -> bytes:
    # Visible binary mask: 0 → 0, 1 → 255
    vis = (mask.astype(np.uint8) * 255)
    image = Image.fromarray(vis, mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def encode_png_probability(probs: np.ndarray) -> bytes:
    vis = (np.clip(probs, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    image = Image.fromarray(vis, mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
