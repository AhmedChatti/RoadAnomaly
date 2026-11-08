"""Experiment E0 — trivial all-background predictor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AllBackgroundModel:
    """Predicts background (0) for every pixel. No trainable parameters."""

    name: str = "E0_all_background"
    threshold: float = 0.5

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 4:
            raise ValueError(f"Expected NHWC batch, got shape {images.shape}")
        n, h, w, _ = images.shape
        return np.zeros((n, h, w, 1), dtype=np.float32)

    def predict(self, images: np.ndarray, threshold: float | None = None) -> np.ndarray:
        thr = self.threshold if threshold is None else threshold
        return (self.predict_proba(images) >= thr).astype(np.uint8)


def predict_all_background(batch_shape: tuple[int, int, int, int]) -> np.ndarray:
    n, h, w, _ = batch_shape
    return np.zeros((n, h, w, 1), dtype=np.float32)
