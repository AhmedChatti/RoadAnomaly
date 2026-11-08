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
    """Official E2b prediction at the frozen operational threshold (0.55)."""

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


@dataclass(frozen=True)
class ExplorationView:
    """Visual-only binary views derived from an existing probability map.

    Not a validated scientific result. Must never replace InferenceResult.
    """

    threshold: float
    binary_mask: np.ndarray
    binary_full: np.ndarray
    overlay_rgb: np.ndarray
    anomaly_pixel_pct: float
    label: str = "Seuil exploratoire — hors protocole scientifique"


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


def _mask_edges(binary_mask: np.ndarray) -> np.ndarray:
    """Simple 4-neighbour edge map for contour overlay (no SciPy dependency)."""
    mask = binary_mask.astype(bool)
    up = np.zeros_like(mask)
    down = np.zeros_like(mask)
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    up[1:, :] = mask[:-1, :]
    down[:-1, :] = mask[1:, :]
    left[:, 1:] = mask[:, :-1]
    right[:, :-1] = mask[:, 1:]
    return mask & ~(up & down & left & right)


def make_overlay(
    image_rgb: np.ndarray,
    binary_mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (232, 93, 42),
    alpha: float = 0.42,
    draw_contour: bool = True,
    contour_color: tuple[int, int, int] = (255, 236, 179),
) -> np.ndarray:
    if image_rgb.shape[:2] != binary_mask.shape[:2]:
        raise ValueError("Overlay: image and mask spatial sizes must match")
    out = image_rgb.astype(np.float32).copy()
    mask = binary_mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    if draw_contour and mask.any():
        edges = _mask_edges(binary_mask.astype(np.uint8))
        out[edges] = np.array(contour_color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def colorize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Readable binary mask visualization (dark asphalt + amber anomaly)."""
    h, w = binary_mask.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (28, 32, 38)
    canvas[binary_mask.astype(bool)] = (232, 93, 42)
    return canvas


def probability_heatmap_rgb(probs: np.ndarray) -> np.ndarray:
    """Map [0,1] probabilities to a warm amber heatmap for display only."""
    p = np.clip(probs.astype(np.float32), 0.0, 1.0)
    # Dark → amber → near-white highlights (no matplotlib colormap dependency).
    r = (40 + 215 * p).round().astype(np.uint8)
    g = (36 + 90 * p).round().astype(np.uint8)
    b = (42 + 20 * (1.0 - p)).round().astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def resize_probability_bilinear(probs: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize HxW probability map to original size (display only; not used for threshold)."""
    # Threshold is applied at model resolution; this map is for visualization / download.
    pil = Image.fromarray((np.clip(probs, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    resized = pil.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


EXPLORATION_LABEL = "Seuil exploratoire — hors protocole scientifique"
EXPLORATION_THRESHOLD_MIN = 0.10
EXPLORATION_THRESHOLD_MAX = 0.90


def _threshold_probability_map(
    probability_map: np.ndarray,
    *,
    threshold: float,
    original_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold at model resolution, then nearest-neighbour resize to original HxW."""
    binary = threshold_probabilities(probability_map, float(threshold)).astype(np.uint8)
    orig_h, orig_w = original_hw
    binary_full = resize_mask_nearest(binary, orig_w, orig_h)
    binary_full = (binary_full > 0).astype(np.uint8)
    return binary, binary_full


def predict_e2b(model: Any, image_rgb: np.ndarray) -> InferenceResult:
    """Run E2b once and produce the **official** prediction at frozen threshold 0.55.

    The operational threshold is hard-coded to ``E2B_THRESHOLD``; callers cannot
    override it. Exploratory thresholds must use ``derive_exploratory_view``.
    """
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
    binary, binary_full = _threshold_probability_map(
        probs, threshold=E2B_THRESHOLD, original_hw=(orig_h, orig_w)
    )
    prob_full = resize_probability_bilinear(probs, orig_w, orig_h)
    overlay = make_overlay(image_rgb, binary_full)
    return InferenceResult(
        original_rgb=image_rgb,
        model_rgb=model_rgb,
        probability_map=probs,
        binary_mask=binary,
        probability_full=prob_full,
        binary_full=binary_full,
        overlay_rgb=overlay,
        anomaly_pixel_pct=float(binary_full.mean() * 100.0),
        mean_anomaly_probability=float(probs.mean()),
        threshold=float(E2B_THRESHOLD),
        input_width=E2B_INPUT_WIDTH,
        input_height=E2B_INPUT_HEIGHT,
    )


def derive_exploratory_view(
    official: InferenceResult,
    exploration_threshold: float,
) -> ExplorationView:
    """Re-threshold an existing official probability map for visual exploration only.

    Does not run the model again and does not mutate ``official``.
    """
    thr = float(exploration_threshold)
    if thr < EXPLORATION_THRESHOLD_MIN or thr > EXPLORATION_THRESHOLD_MAX:
        raise ValueError(
            f"Seuil exploratoire hors plage "
            f"[{EXPLORATION_THRESHOLD_MIN}, {EXPLORATION_THRESHOLD_MAX}]: {thr}"
        )
    orig_h, orig_w = official.original_rgb.shape[:2]
    # Defensive copy so exploration cannot alias/mutate official arrays.
    probs = np.array(official.probability_map, copy=True)
    binary, binary_full = _threshold_probability_map(
        probs, threshold=thr, original_hw=(orig_h, orig_w)
    )
    overlay = make_overlay(official.original_rgb, binary_full)
    return ExplorationView(
        threshold=thr,
        binary_mask=binary,
        binary_full=binary_full,
        overlay_rgb=overlay,
        anomaly_pixel_pct=float(binary_full.mean() * 100.0),
        label=EXPLORATION_LABEL,
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
