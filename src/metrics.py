"""Binary segmentation metrics (pixel-level).

Conventions follow docs/02_protocole_scientifique_phase1.md §7.3.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_binary(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    true = _as_binary(y_true).ravel()
    pred = _as_binary(y_pred).ravel()
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: true {true.shape} vs pred {pred.shape}")
    tp = int(np.logical_and(true == 1, pred == 1).sum())
    fp = int(np.logical_and(true == 0, pred == 1).sum())
    fn = int(np.logical_and(true == 1, pred == 0).sum())
    tn = int(np.logical_and(true == 0, pred == 0).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp + fp == 0:
        return 1.0 if tp + fn == 0 else 0.0
    return tp / (tp + fp)


def recall_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp + fn == 0:
        return 1.0 if tp + fp == 0 else 0.0
    return tp / (tp + fn)


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    p = precision_from_counts(tp, fp, fn)
    r = recall_from_counts(tp, fp, fn)
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return tp / denom


def dice_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return (2 * tp) / denom


def accuracy_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    total = tp + fp + fn + tn
    if total == 0:
        return 1.0
    return (tp + tn) / total


def binary_segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    counts = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    return {
        **counts,
        "precision": precision_from_counts(tp, fp, fn),
        "recall": recall_from_counts(tp, fp, fn),
        "f1": f1_from_counts(tp, fp, fn),
        "iou": iou_from_counts(tp, fp, fn),
        "dice": dice_from_counts(tp, fp, fn),
        "accuracy": accuracy_from_counts(tp, fp, fn, tn),
    }


def threshold_probabilities(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probs) >= threshold).astype(np.uint8)
