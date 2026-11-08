"""Training-only synchronized image/mask augmentation.

Geometric transforms always use nearest-neighbour on the mask.
Photometric transforms never modify the mask.
Vertical flips are disabled by protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from config import load_config
from preprocessing import MODEL_ANOMALY, remap_official_mask_to_binary


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_degrees_max: float = 8.0
    zoom_range: float = 0.1
    translation_fraction: float = 0.05
    brightness_delta: float = 0.15
    contrast_range: float = 0.15
    p_horizontal_flip: float = 0.5
    p_rotation: float = 0.5
    p_affine: float = 0.5
    p_brightness_contrast: float = 0.5

    @classmethod
    def from_project_config(cls, config: dict | None = None) -> "AugmentationConfig":
        cfg = (config or load_config())["augmentation"]
        return cls(
            horizontal_flip=bool(cfg.get("horizontal_flip", True)),
            vertical_flip=bool(cfg.get("vertical_flip", False)),
            rotation_degrees_max=float(cfg.get("rotation_degrees_max", 8)),
            zoom_range=float(cfg.get("zoom_range", 0.1)),
            translation_fraction=float(cfg.get("translation_fraction", 0.05)),
            brightness_delta=float(cfg.get("brightness_delta", 0.15)),
            contrast_range=float(cfg.get("contrast_range", 0.15)),
            p_horizontal_flip=float(cfg.get("p_horizontal_flip", 0.5)),
            p_rotation=float(cfg.get("p_rotation", 0.5)),
            p_affine=float(cfg.get("p_affine", 0.5)),
            p_brightness_contrast=float(cfg.get("p_brightness_contrast", 0.5)),
        )


def _as_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.min() < 0 or image.max() > 1.5:
        # Likely ImageNet-normalized; augmentation expects display-space [0,1] or uint8.
        raise ValueError(
            "Augment images in [0,1] or uint8 before ImageNet normalization, "
            "or pass image_uint8 from preprocess_pair."
        )
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _to_float01(image_uint8: np.ndarray) -> np.ndarray:
    return image_uint8.astype(np.float32) / 255.0


def _affine_pair(
    image_uint8: np.ndarray,
    mask: np.ndarray,
    *,
    angle: float,
    scale: float,
    translate_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate, zoom and translate while keeping the canvas size fixed.

    PIL's AFFINE transform maps each output pixel to an input coordinate:
    x_in = a*x_out + b*y_out + c
    y_in = d*x_out + e*y_out + f
    """
    height, width = image_uint8.shape[:2]
    cx, cy = width / 2.0, height / 2.0
    tx, ty = translate_xy
    safe_scale = scale if abs(scale) > 1e-6 else 1.0

    image_pil = Image.fromarray(image_uint8, mode="RGB")
    mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L")
    image_r = image_pil.rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(0, 0, 0)
    )
    mask_r = mask_pil.rotate(
        angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0
    )

    inv = 1.0 / safe_scale
    coeffs = (
        inv,
        0.0,
        cx - inv * (cx + tx),
        0.0,
        inv,
        cy - inv * (cy + ty),
    )
    image_a = image_r.transform(
        (width, height),
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    mask_a = mask_r.transform(
        (width, height),
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
    )
    return (
        np.asarray(image_a, dtype=np.uint8),
        remap_official_mask_to_binary(np.asarray(mask_a, dtype=np.uint8)),
    )


def _brightness_contrast(image_uint8: np.ndarray, rng: np.random.Generator, cfg: AugmentationConfig) -> np.ndarray:
    image = _to_float01(image_uint8)
    brightness = 1.0 + float(rng.uniform(-cfg.brightness_delta, cfg.brightness_delta))
    contrast = 1.0 + float(rng.uniform(-cfg.contrast_range, cfg.contrast_range))
    mean = image.mean(axis=(0, 1), keepdims=True)
    image = (image - mean) * contrast + mean
    image = image * brightness
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    config: AugmentationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random augmentation. Returns uint8 image and binary {0,1} mask."""
    cfg = config or AugmentationConfig.from_project_config()
    if cfg.vertical_flip:
        raise ValueError("Vertical flip is disabled by the Phase 1 protocol.")
    rng = rng or np.random.default_rng()
    image_u8 = _as_uint8_image(image)
    mask_bin = remap_official_mask_to_binary(mask)

    if cfg.horizontal_flip and rng.random() < cfg.p_horizontal_flip:
        image_u8 = np.ascontiguousarray(image_u8[:, ::-1, :])
        mask_bin = np.ascontiguousarray(mask_bin[:, ::-1])

    angle = 0.0
    if cfg.rotation_degrees_max > 0 and rng.random() < cfg.p_rotation:
        angle = float(rng.uniform(-cfg.rotation_degrees_max, cfg.rotation_degrees_max))

    scale = 1.0
    tx = 0.0
    ty = 0.0
    if rng.random() < cfg.p_affine:
        scale = float(rng.uniform(1.0 - cfg.zoom_range, 1.0 + cfg.zoom_range))
        height, width = image_u8.shape[:2]
        tx = float(rng.uniform(-cfg.translation_fraction, cfg.translation_fraction) * width)
        ty = float(rng.uniform(-cfg.translation_fraction, cfg.translation_fraction) * height)

    if angle != 0.0 or scale != 1.0 or tx != 0.0 or ty != 0.0:
        image_u8, mask_bin = _affine_pair(
            image_u8,
            mask_bin,
            angle=angle,
            scale=scale,
            translate_xy=(tx, ty),
        )

    if mask_bin.sum() == 0:
        # Degenerate crop/zoom — return original pair rather than an empty label.
        return _as_uint8_image(image), remap_official_mask_to_binary(mask)

    if rng.random() < cfg.p_brightness_contrast:
        image_u8 = _brightness_contrast(image_u8, rng, cfg)

    if set(np.unique(mask_bin).tolist()) - {0, 1}:
        raise RuntimeError("Augmentation produced non-binary mask labels")
    if not np.any(mask_bin == MODEL_ANOMALY):
        return _as_uint8_image(image), remap_official_mask_to_binary(mask)
    return image_u8, mask_bin
