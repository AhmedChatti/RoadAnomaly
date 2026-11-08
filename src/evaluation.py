"""Validation/evaluation helpers. Do not use the test split for model selection."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from dataset import RoadAnomalyDataset, SplitName
from metrics import binary_segmentation_metrics, threshold_probabilities
from preprocessing import NormalizeMode


def evaluate_predictor_on_split(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    split: SplitName,
    *,
    threshold: float = 0.5,
    normalize: NormalizeMode = "scale_0_1",
    batch_size: int = 4,
    allow_test: bool = False,
) -> dict[str, Any]:
    """Evaluate a probability predictor on one split.

    `allow_test` must stay False during Phase 3 model selection.
    """
    if split == "test" and not allow_test:
        raise RuntimeError(
            "Test-split evaluation is blocked until the final model and threshold are frozen "
            "(Phase 6). Pass allow_test=True only then."
        )

    data = RoadAnomalyDataset(split, normalize=normalize, augment=False)
    per_image = []
    global_tp = global_fp = global_fn = global_tn = 0

    for start in range(0, len(data), batch_size):
        batch_records = []
        images = []
        masks = []
        for offset in range(batch_size):
            index = start + offset
            if index >= len(data):
                break
            sample = data[index]
            batch_records.append(sample)
            images.append(sample["image"])
            masks.append(sample["mask_uint8"])
        x = np.stack(images, axis=0)
        probs = predict_proba_fn(x)
        if probs.shape[-1] == 1:
            prob_maps = probs[..., 0]
        else:
            prob_maps = probs
        preds = threshold_probabilities(prob_maps, threshold)

        for i, sample in enumerate(batch_records):
            metrics = binary_segmentation_metrics(masks[i], preds[i])
            per_image.append(
                {
                    "frame_id": sample["frame_id"],
                    "category": sample["category"],
                    "split": split,
                    "anomaly_ratio": sample["anomaly_ratio"],
                    **{k: metrics[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou", "dice", "accuracy")},
                }
            )
            global_tp += metrics["tp"]
            global_fp += metrics["fp"]
            global_fn += metrics["fn"]
            global_tn += metrics["tn"]

    table = pd.DataFrame(per_image)
    from metrics import (
        accuracy_from_counts,
        dice_from_counts,
        f1_from_counts,
        iou_from_counts,
        precision_from_counts,
        recall_from_counts,
    )

    global_metrics = {
        "tp": global_tp,
        "fp": global_fp,
        "fn": global_fn,
        "tn": global_tn,
        "precision": precision_from_counts(global_tp, global_fp, global_fn),
        "recall": recall_from_counts(global_tp, global_fp, global_fn),
        "f1": f1_from_counts(global_tp, global_fp, global_fn),
        "iou": iou_from_counts(global_tp, global_fp, global_fn),
        "dice": dice_from_counts(global_tp, global_fp, global_fn),
        "accuracy": accuracy_from_counts(global_tp, global_fp, global_fn, global_tn),
    }
    mean_metrics = {
        "mean_precision": float(table["precision"].mean()),
        "mean_recall": float(table["recall"].mean()),
        "mean_f1": float(table["f1"].mean()),
        "mean_iou": float(table["iou"].mean()),
        "mean_dice": float(table["dice"].mean()),
        "mean_accuracy": float(table["accuracy"].mean()),
    }
    return {
        "split": split,
        "threshold": threshold,
        "n_images": len(table),
        "global": global_metrics,
        "mean_per_image": mean_metrics,
        "per_image": table,
    }


def select_threshold_on_validation(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    thresholds: list[float] | None = None,
    *,
    normalize: NormalizeMode = "scale_0_1",
    batch_size: int = 4,
) -> dict[str, Any]:
    thresholds = thresholds or [round(x, 2) for x in np.linspace(0.1, 0.9, 17)]
    rows = []
    best = None
    for thr in thresholds:
        result = evaluate_predictor_on_split(
            predict_proba_fn,
            "val",
            threshold=float(thr),
            normalize=normalize,
            batch_size=batch_size,
            allow_test=False,
        )
        score = result["mean_per_image"]["mean_dice"]
        rows.append({"threshold": float(thr), "val_mean_dice": score, "val_mean_iou": result["mean_per_image"]["mean_iou"]})
        if best is None or score > best["val_mean_dice"]:
            best = {
                "threshold": float(thr),
                "val_mean_dice": score,
                "result": result,
            }
    assert best is not None
    return {
        "best_threshold": best["threshold"],
        "best_val_mean_dice": best["val_mean_dice"],
        "grid": pd.DataFrame(rows),
        "best_result": best["result"],
    }
