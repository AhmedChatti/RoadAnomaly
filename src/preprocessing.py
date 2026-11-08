"""Deterministic image/mask preprocessing for training, evaluation and inference.

Protocol (Phase 1):
- Resize to 512 x 288 without letterbox padding.
- Remap official labels {0, 2} to model labels {0, 1}.
- Bilinear resize for RGB images; nearest-neighbour for masks.
- Normalization modes: scale_0_1 or imagenet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from config import load_config, preprocessing_size

NormalizeMode = Literal["scale_0_1", "imagenet", "none"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OFFICIAL_BACKGROUND = 0
OFFICIAL_ANOMALY = 2
MODEL_BACKGROUND = 0
MODEL_ANOMALY = 1


def load_rgb_image(path: Path | str) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image at {path}, got shape {array.shape}")
    return array


def load_semantic_mask(path: Path | str) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        # Some tools save duplicated channels; reduce to a single channel.
        if array.shape[-1] in (3, 4) and np.all(array[..., 0] == array[..., 1]):
            array = array[..., 0]
        else:
            raise ValueError(f"Unexpected multi-channel mask at {path}: {array.shape}")
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D mask at {path}, got shape {array.shape}")
    return array.astype(np.uint8, copy=False)


def assert_image_mask_alignment(image: np.ndarray, mask: np.ndarray, frame_id: str = "") -> None:
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image/mask size mismatch{f' for {frame_id}' if frame_id else ''}: "
            f"image {image.shape[:2]} vs mask {mask.shape[:2]}"
        )


def remap_official_mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """Map official RoadAnomaly labels to {0, 1}.

    Accepted input values: 0 (background) and 2 (anomaly).
    Also accepts already-prepared masks that use 255 or 1 for the positive class.
    """
    flat_unique = np.unique(mask)
    allowed_official = {OFFICIAL_BACKGROUND, OFFICIAL_ANOMALY}
    allowed_prepared = {0, 1, 255}
    values = set(int(v) for v in flat_unique)
    if not values.issubset(allowed_official | allowed_prepared):
        raise ValueError(f"Unexpected mask values {sorted(values)}; expected {{0,2}} or {{0,1/255}}")

    binary = np.zeros(mask.shape, dtype=np.uint8)
    positive = (mask == OFFICIAL_ANOMALY) | (mask == 255) | (mask == MODEL_ANOMALY)
    binary[positive] = MODEL_ANOMALY
    return binary


def resize_image_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray(image, mode="RGB")
    resized = pil.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def resize_mask_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray(mask, mode="L")
    resized = pil.resize((width, height), resample=Image.Resampling.NEAREST)
    out = np.asarray(resized, dtype=np.uint8)
    # Nearest can only produce existing labels; still force binary {0,1}.
    return remap_official_mask_to_binary(out)


def normalize_image(image_uint8: np.ndarray, mode: NormalizeMode) -> np.ndarray:
    image = image_uint8.astype(np.float32) / 255.0
    if mode == "none":
        return image
    if mode == "scale_0_1":
        return image
    if mode == "imagenet":
        return (image - IMAGENET_MEAN) / IMAGENET_STD
    raise ValueError(f"Unknown normalize mode: {mode}")


def denormalize_image(image_float: np.ndarray, mode: NormalizeMode) -> np.ndarray:
    """Convert a normalized float image back to uint8 RGB for visualization."""
    array = image_float.astype(np.float32)
    if mode == "imagenet":
        array = array * IMAGENET_STD + IMAGENET_MEAN
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).round().astype(np.uint8)


def preprocess_pair(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    width: int | None = None,
    height: int | None = None,
    normalize: NormalizeMode = "scale_0_1",
    frame_id: str = "",
) -> dict:
    """Full deterministic preprocess used by train/val/test/inference."""
    assert_image_mask_alignment(image, mask, frame_id=frame_id)
    binary = remap_official_mask_to_binary(mask)
    if binary.sum() == 0:
        raise ValueError(f"Mask has no anomaly pixels{f' ({frame_id})' if frame_id else ''}")

    cfg_h, cfg_w = preprocessing_size(load_config())
    target_h = height if height is not None else cfg_h
    target_w = width if width is not None else cfg_w

    image_r = resize_image_rgb(image, target_w, target_h)
    mask_r = resize_mask_nearest(binary, target_w, target_h)
    if set(np.unique(mask_r).tolist()) - {0, 1}:
        raise RuntimeError(f"Mask resize produced invalid labels for {frame_id or 'frame'}")
    if mask_r.sum() == 0:
        raise RuntimeError(f"Resize erased all anomaly pixels for {frame_id or 'frame'}")

    image_n = normalize_image(image_r, normalize)
    return {
        "image": image_n.astype(np.float32),
        "mask": mask_r.astype(np.uint8),
        "image_uint8": image_r,
        "height": target_h,
        "width": target_w,
        "normalize": normalize,
        "anomaly_pixels": int(mask_r.sum()),
        "anomaly_ratio": float(mask_r.mean()),
    }


def preprocess_paths(
    image_path: Path | str,
    mask_path: Path | str,
    *,
    normalize: NormalizeMode = "scale_0_1",
    frame_id: str = "",
) -> dict:
    image = load_rgb_image(image_path)
    mask = load_semantic_mask(mask_path)
    return preprocess_pair(image, mask, normalize=normalize, frame_id=frame_id)
