"""Regenerate Phase 3 E1 validation qualitative panel from the canonical checkpoint.

SAFETY:
- Uses only the validation split.
- Does not access the test set.
- Does not retrain or alter frozen metrics / metadata / checkpoint.
- Overwrites only experiments/figures/phase3_E1_val_qualitative.png (visualization).
- Writes a sidecar provenance JSON documenting that the figure is regenerated.

This is a Stage 1 recovery action for a figure lost during local disk cleanup.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import RoadAnomalyDataset  # noqa: E402
from evaluation import evaluate_predictor_on_split  # noqa: E402
from losses import bce_dice_loss  # noqa: E402
from reproducibility import EXPECTED_PHASE3_CHECKPOINT_SHA256, sha256_file  # noqa: E402

CANONICAL = "E1_cnn_scratch_20261108_1510"
CKPT = ROOT / "checkpoints" / CANONICAL / "best.keras"
META = ROOT / "experiments" / "metadata" / f"{CANONICAL}.json"
OUT_PNG = ROOT / "experiments" / "figures" / "phase3_E1_val_qualitative.png"
OUT_SIDECAR = ROOT / "experiments" / "figures" / "phase3_E1_val_qualitative.REGENERATED.json"
VAL_THR_DEFAULT = 0.35


def _overlay(image: np.ndarray, mask: np.ndarray, color=(220, 30, 45), alpha=0.45) -> np.ndarray:
    painted = image.copy()
    region = mask > 0
    if region.any():
        tint = np.array(color, dtype=np.float32)
        pixels = painted[region].astype(np.float32)
        painted[region] = np.clip(pixels * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
    return painted


def _error_map(true_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    true = true_mask > 0
    pred = pred_mask > 0
    out = np.zeros((*true.shape, 3), dtype=np.uint8)
    out[true & pred] = (40, 180, 40)
    out[~true & pred] = (220, 40, 40)
    out[true & ~pred] = (40, 90, 220)
    return out


def main() -> int:
    from tensorflow import keras

    if not CKPT.is_file():
        print(f"FAIL missing canonical checkpoint: {CKPT}")
        return 1
    ckpt_sha = sha256_file(CKPT)
    if ckpt_sha != EXPECTED_PHASE3_CHECKPOINT_SHA256:
        print("FAIL checkpoint SHA mismatch; refusing to regenerate from non-canonical weights")
        print(f"  got      {ckpt_sha}")
        print(f"  expected {EXPECTED_PHASE3_CHECKPOINT_SHA256}")
        return 1

    meta = json.loads(META.read_text(encoding="utf-8"))
    best_thr = float(meta["threshold_selection"]["value"])
    if abs(best_thr - VAL_THR_DEFAULT) > 1e-9:
        print(f"WARN metadata threshold={best_thr} differs from documented 0.35; using metadata value")

    model = keras.models.load_model(
        CKPT,
        custom_objects={"bce_dice": bce_dice_loss(positive_weight=15.0)},
        compile=False,
    )

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        return model.predict(batch, verbose=0)

    val = evaluate_predictor_on_split(predict_fn, "val", threshold=best_thr, allow_test=False)
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
        axes[row, 0].set_title(frame_id, fontsize=8)
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
        f"E1 validation qualitative panel (REGENERATED) — threshold={best_thr:.2f} — green TP, red FP, blue FN",
        fontsize=11,
    )
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, facecolor="white")
    plt.close(fig)

    png_sha = hashlib.sha256(OUT_PNG.read_bytes()).hexdigest()
    sidecar = {
        "artefact": "phase3_E1_val_qualitative.png",
        "status": "regenerated",
        "regenerated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(CKPT.relative_to(ROOT)).replace("\\", "/"),
        "source_checkpoint_sha256": ckpt_sha,
        "split": "val",
        "threshold": best_thr,
        "selection_rule": "worst Dice / best Dice / smallest anomaly_ratio on validation per-image table",
        "picked_frame_ids": [{"role": role, "frame_id": fid} for role, fid in picks],
        "val_mean_dice_at_threshold": float(val["mean_per_image"]["mean_dice"]),
        "frozen_val_mean_dice": float(meta["validation_metrics"]["mean_per_image"]["mean_dice"]),
        "png_sha256": png_sha,
        "notes": (
            "Figure was missing after local disk cleanup. Regenerated from the canonical "
            "E1_1510 checkpoint and validation split only. Frozen experiment metrics/metadata "
            "were not modified. This PNG is a recovered visualization, not a new experiment."
        ),
    }
    OUT_SIDECAR.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PNG.relative_to(ROOT)}")
    print(f"Wrote {OUT_SIDECAR.relative_to(ROOT)}")
    print(f"png_sha256={png_sha}")
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
