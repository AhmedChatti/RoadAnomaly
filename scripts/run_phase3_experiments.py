"""Phase 3: E0 all-background baseline + E1 scratch FCN.

Evaluates on validation only for model selection / reporting in this phase.
Does not tune on the test set.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config, project_root  # noqa: E402
from dataset import RoadAnomalyDataset  # noqa: E402
from evaluation import evaluate_predictor_on_split, select_threshold_on_validation  # noqa: E402
from models.baseline import AllBackgroundModel  # noqa: E402
from training import software_versions, train_scratch_fcn, write_metadata  # noqa: E402


def _overlay(image: np.ndarray, mask: np.ndarray, color=(220, 30, 45), alpha=0.45) -> np.ndarray:
    painted = image.copy()
    region = mask > 0
    if region.any():
        tint = np.array(color, dtype=np.float32)
        pixels = painted[region].astype(np.float32)
        painted[region] = np.clip(pixels * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
    return painted


def _error_map(true_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """RGB error viz: green=TP, red=FP, blue=FN, black=TN."""
    true = true_mask > 0
    pred = pred_mask > 0
    h, w = true.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[true & pred] = (40, 180, 40)
    out[~true & pred] = (220, 40, 40)
    out[true & ~pred] = (40, 90, 220)
    return out


def run_e0(results_dir: Path, figures_dir: Path) -> dict:
    cfg = load_config()
    model = AllBackgroundModel()
    experiment_id = "E0_all_background"

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(x)

    val = evaluate_predictor_on_split(predict_fn, "val", threshold=0.5, allow_test=False)
    train = evaluate_predictor_on_split(predict_fn, "train", threshold=0.5, allow_test=False)

    meta = {
        "experiment_id": experiment_id,
        "model_name": "all_background",
        "architecture": "Constant zero probability map",
        "input_height": cfg["preprocessing"]["input_height"],
        "input_width": cfg["preprocessing"]["input_width"],
        "split_manifest": "data/processed/split_manifest.csv",
        "split_manifest_sha256": cfg["split"]["assignment_sha256"],
        "seed": cfg["training"]["seed"],
        "optimizer": None,
        "learning_rate": None,
        "batch_size": None,
        "epochs_planned": 0,
        "epochs_run": 0,
        "loss": None,
        "metrics_monitored": ["precision", "recall", "f1", "iou", "dice", "accuracy"],
        "augmentation": None,
        "checkpoint_selection_criterion": None,
        "best_epoch": None,
        "best_val_metric_name": "val_mean_dice",
        "best_val_metric_value": val["mean_per_image"]["mean_dice"],
        "threshold_selection": {"split": "val", "metric": "fixed", "value": 0.5},
        "hardware": "n/a",
        "software_versions": software_versions(),
        "duration_seconds": 0.0,
        "checkpoint_path": None,
        "validation_metrics": {
            "global": val["global"],
            "mean_per_image": val["mean_per_image"],
        },
        "train_metrics_for_reference": {
            "global": train["global"],
            "mean_per_image": train["mean_per_image"],
        },
        "notes": (
            "Trivial baseline: predict background everywhere. "
            "Expect near-zero anomaly IoU/Dice and high pixel accuracy."
        ),
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_metadata(ROOT / "experiments" / "metadata" / f"{experiment_id}.json", meta)
    val["per_image"].to_csv(results_dir / f"{experiment_id}_val_per_image.csv", index=False)

    # Accuracy trap figure
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["Accuracy", "IoU", "Dice", "Recall"]
    values = [
        val["global"]["accuracy"] * 100,
        val["global"]["iou"] * 100,
        val["global"]["dice"] * 100,
        val["global"]["recall"] * 100,
    ]
    bars = ax.bar(labels, values, color=["#B8B8B8", "#E45756", "#F58518", "#4C78A8"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center")
    ax.set_ylim(0, 110)
    ax.set_ylabel("%")
    ax.set_title("E0 all-background on validation: accuracy is misleading")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_E0_metric_trap.png", dpi=140, facecolor="white")
    plt.close(fig)
    return meta


def run_e1(results_dir: Path, figures_dir: Path) -> dict:
    experiment_id = f"E1_cnn_scratch_{datetime.now().strftime('%Y%m%d_%H%M')}"
    train_out = train_scratch_fcn(experiment_id=experiment_id)
    model = train_out["model"]
    meta = train_out["metadata"]

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return model.predict(x, verbose=0)

    # Threshold selection on validation only.
    thr_info = select_threshold_on_validation(predict_fn, normalize="scale_0_1", batch_size=4)
    thr_info["grid"].to_csv(results_dir / f"{experiment_id}_val_threshold_grid.csv", index=False)
    best_thr = thr_info["best_threshold"]
    val = evaluate_predictor_on_split(
        predict_fn, "val", threshold=best_thr, allow_test=False
    )
    val["per_image"].to_csv(results_dir / f"{experiment_id}_val_per_image.csv", index=False)

    meta["threshold_selection"] = {
        "split": "val",
        "metric": "mean_dice",
        "value": best_thr,
        "grid_best_val_mean_dice": thr_info["best_val_mean_dice"],
    }
    meta["validation_metrics"] = {
        "global": val["global"],
        "mean_per_image": val["mean_per_image"],
        "threshold": best_thr,
    }
    write_metadata(train_out["metadata_path"], meta)

    # Training curves
    hist = meta["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    epochs = np.arange(1, len(hist["loss"]) + 1)
    axes[0].plot(epochs, hist["loss"], label="train loss")
    axes[0].plot(epochs, hist["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].legend(frameon=False)
    axes[0].set_xlabel("Epoch")
    axes[1].plot(np.arange(1, len(hist["val_mean_dice"]) + 1), hist["val_mean_dice"], color="#54A24B")
    axes[1].axvline(meta["best_epoch"], color="#E45756", linestyle="--", label=f"best @{meta['best_epoch']}")
    axes[1].set_title("Validation mean Dice / image")
    axes[1].legend(frameon=False)
    axes[1].set_xlabel("Epoch")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"{experiment_id} training curves")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_E1_training_curves.png", dpi=140, facecolor="white")
    plt.close(fig)

    # Qualitative val panel: best / worst / small anomaly
    data = RoadAnomalyDataset("val", normalize="scale_0_1", augment=False)
    table = val["per_image"].sort_values("dice")
    picks = [
        ("worst Dice", table.iloc[0]["frame_id"]),
        ("best Dice", table.iloc[-1]["frame_id"]),
        ("smallest anomaly", table.sort_values("anomaly_ratio").iloc[0]["frame_id"]),
    ]
    id_to_index = {rec.frame_id: i for i, rec in enumerate(data.records)}
    fig, axes = plt.subplots(len(picks), 4, figsize=(12.5, 9.0))
    for row, (title, frame_id) in enumerate(picks):
        sample = data[id_to_index[frame_id]]
        prob = model.predict(sample["image"][None, ...], verbose=0)[0, ..., 0]
        pred = (prob >= best_thr).astype(np.uint8)
        true = sample["mask_uint8"]
        axes[row, 0].imshow(sample["image_uint8"])
        axes[row, 0].set_ylabel(title, fontsize=10)
        axes[row, 0].set_title(frame_id if row == 0 else frame_id, fontsize=8)
        axes[row, 1].imshow(true, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("GT" if row == 0 else "")
        axes[row, 2].imshow(_overlay(sample["image_uint8"], pred))
        axes[row, 2].set_title("Prediction" if row == 0 else "")
        axes[row, 3].imshow(_error_map(true, pred))
        axes[row, 3].set_title("TP/FP/FN" if row == 0 else "")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"E1 validation qualitative panel (threshold={best_thr:.2f}) — green TP, red FP, blue FN",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_E1_val_qualitative.png", dpi=130, facecolor="white")
    plt.close(fig)

    # Comparison table figure E0 vs E1
    e0_path = ROOT / "experiments" / "metadata" / "E0_all_background.json"
    e0 = json.loads(e0_path.read_text(encoding="utf-8"))
    rows = [
        {
            "experiment": "E0",
            "mean_dice": e0["validation_metrics"]["mean_per_image"]["mean_dice"],
            "mean_iou": e0["validation_metrics"]["mean_per_image"]["mean_iou"],
            "mean_recall": e0["validation_metrics"]["mean_per_image"]["mean_recall"],
            "global_accuracy": e0["validation_metrics"]["global"]["accuracy"],
        },
        {
            "experiment": "E1",
            "mean_dice": val["mean_per_image"]["mean_dice"],
            "mean_iou": val["mean_per_image"]["mean_iou"],
            "mean_recall": val["mean_per_image"]["mean_recall"],
            "global_accuracy": val["global"]["accuracy"],
        },
    ]
    pd.DataFrame(rows).to_csv(results_dir / "phase3_E0_E1_val_comparison.csv", index=False)
    return meta


def main() -> None:
    root = project_root()
    results_dir = root / "experiments" / "results"
    figures_dir = root / "experiments" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== E0 all-background baseline ===", flush=True)
    e0 = run_e0(results_dir, figures_dir)
    print(json.dumps(e0["validation_metrics"], indent=2), flush=True)

    print("=== E1 scratch FCN training ===", flush=True)
    e1 = run_e1(results_dir, figures_dir)
    print(
        json.dumps(
            {
                "experiment_id": e1["experiment_id"],
                "best_epoch": e1["best_epoch"],
                "best_val_mean_dice_during_train": e1["best_val_metric_value"],
                "selected_threshold": e1["threshold_selection"]["value"],
                "validation_metrics": e1["validation_metrics"],
            },
            indent=2,
        ),
        flush=True,
    )
    print("Phase 3 complete. Test split was not used for selection.", flush=True)


if __name__ == "__main__":
    main()
