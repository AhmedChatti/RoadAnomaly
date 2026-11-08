"""Phase 3 freeze audit: reload E1 checkpoint, predict on val only, verify thresholds.

Does not access the test split. Does not retrain. Does not overwrite Phase 3 metrics.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config, project_root  # noqa: E402
from dataset import RoadAnomalyDataset, assignment_sha256_from_manifest, load_split_manifest  # noqa: E402
from evaluation import evaluate_predictor_on_split  # noqa: E402
from losses import bce_dice_loss  # noqa: E402


CANONICAL_E1 = "E1_cnn_scratch_20261108_1510"
CANONICAL_E0 = "E0_all_background"
TRAIN_THR = 0.5
VAL_THR = 0.35


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str, issues: list[str]) -> None:
    if condition:
        print(f"  OK  {message}")
    else:
        print(f"FAIL  {message}")
        issues.append(message)


def main() -> int:
    import tensorflow as tf
    from tensorflow import keras

    root = project_root()
    issues: list[str] = []
    print("=== Phase 3 freeze audit ===")

    # --- Artifact presence ---
    print("\n[1] Artifacts")
    required_paths = [
        root / "experiments" / "metadata" / f"{CANONICAL_E0}.json",
        root / "experiments" / "metadata" / f"{CANONICAL_E1}.json",
        root / "experiments" / "results" / "phase3_E0_E1_val_comparison.csv",
        root / "experiments" / "results" / f"{CANONICAL_E0}_val_per_image.csv",
        root / "experiments" / "results" / f"{CANONICAL_E1}_val_per_image.csv",
        root / "experiments" / "results" / f"{CANONICAL_E1}_val_threshold_grid.csv",
        root / "experiments" / "figures" / "phase3_E0_metric_trap.png",
        root / "experiments" / "figures" / "phase3_E1_training_curves.png",
        root / "experiments" / "figures" / "phase3_E1_val_qualitative.png",
        root / "checkpoints" / CANONICAL_E1 / "best.keras",
        root / "docs" / "09_phase3_baseline_and_cnn.md",
        root / "src" / "models" / "baseline.py",
        root / "src" / "models" / "cnn.py",
        root / "src" / "losses.py",
        root / "src" / "training.py",
        root / "src" / "evaluation.py",
        root / "scripts" / "run_phase3_experiments.py",
        root / "tests" / "test_dataset.py",
        root / "tests" / "test_metrics.py",
        root / "tests" / "test_preprocessing.py",
    ]
    for path in required_paths:
        require(path.is_file() and path.stat().st_size > 0, f"exists: {path.relative_to(root)}", issues)

    # --- Metadata / thresholds ---
    print("\n[2] Metadata and thresholds")
    e0 = json.loads((root / "experiments" / "metadata" / f"{CANONICAL_E0}.json").read_text(encoding="utf-8"))
    e1 = json.loads((root / "experiments" / "metadata" / f"{CANONICAL_E1}.json").read_text(encoding="utf-8"))
    cfg = load_config()
    manifest = load_split_manifest(cfg)
    split_hash = assignment_sha256_from_manifest(manifest)

    require(e0["status"] == "completed", "E0 status=completed", issues)
    require(e1["status"] == "completed", "E1 status=completed", issues)
    require(e0["split_manifest_sha256"] == split_hash, "E0 split hash matches lock", issues)
    require(e1["split_manifest_sha256"] == split_hash, "E1 split hash matches lock", issues)
    require(e1["input_height"] == 288 and e1["input_width"] == 512, "E1 input 512x288", issues)
    require(e1["best_epoch"] == 66, "E1 best_epoch=66", issues)
    require(abs(float(e1["best_val_metric_value"]) - 0.5631794123253453) < 1e-9, "E1 train-time best Dice @0.5", issues)
    require(e1["threshold_selection"]["split"] == "val", "E1 threshold selected on val", issues)
    require(float(e1["threshold_selection"]["value"]) == VAL_THR, "E1 selected threshold=0.35", issues)
    require(
        abs(float(e1["validation_metrics"]["mean_per_image"]["mean_dice"]) - 0.5689638550170618) < 1e-9,
        "E1 frozen val mean Dice @0.35",
        issues,
    )
    require(
        abs(float(e1["validation_metrics"]["threshold"]) - VAL_THR) < 1e-9,
        "E1 validation_metrics.threshold=0.35",
        issues,
    )

    # Training-time threshold must be recorded separately from selected threshold.
    train_thr = e1["threshold_selection"].get("default_during_training")
    require(
        train_thr is not None and float(train_thr) == TRAIN_THR,
        "E1 metadata records default_during_training=0.5 separately from value=0.35",
        issues,
    )
    require(float(e1["threshold_selection"]["value"]) != float(train_thr or -1), "0.35 != 0.5 in metadata", issues)

    grid_path = root / "experiments" / "results" / f"{CANONICAL_E1}_val_threshold_grid.csv"
    grid_lines = grid_path.read_text(encoding="utf-8").strip().splitlines()
    row_05 = [ln for ln in grid_lines if ln.startswith("0.5,")]
    row_035 = [ln for ln in grid_lines if ln.startswith("0.35,")]
    require(len(row_05) == 1, "threshold grid contains 0.5 row", issues)
    require(len(row_035) == 1, "threshold grid contains 0.35 row", issues)
    if row_05:
        dice_05 = float(row_05[0].split(",")[1])
        require(abs(dice_05 - float(e1["best_val_metric_value"])) < 1e-9, "grid@0.5 matches best_val_metric_value", issues)
    if row_035:
        dice_035 = float(row_035[0].split(",")[1])
        require(
            abs(dice_035 - float(e1["validation_metrics"]["mean_per_image"]["mean_dice"])) < 1e-9,
            "grid@0.35 matches frozen validation mean Dice",
            issues,
        )

    hist = e1.get("history") or {}
    require(
        len(hist.get("loss", [])) == 78 and len(hist.get("val_mean_dice", [])) == 78,
        "E1 history has 78 epochs (loss + val_mean_dice)",
        issues,
    )
    require(abs(max(hist.get("val_mean_dice", [0.0])) - float(e1["best_val_metric_value"])) < 1e-9, "history max Dice = best", issues)

    # --- Checkpoint load + val prediction ---
    print("\n[3] Checkpoint reload + val predictions (no test)")
    ckpt = root / "checkpoints" / CANONICAL_E1 / "best.keras"
    ckpt_hash = sha256_file(ckpt)
    print(f"  INFO checkpoint sha256={ckpt_hash}")
    model = keras.models.load_model(
        ckpt,
        custom_objects={"bce_dice": bce_dice_loss(positive_weight=15.0)},
        compile=False,
    )
    val_data = RoadAnomalyDataset("val", normalize="scale_0_1", augment=False)
    require(len(val_data) == 9, "val split has 9 images", issues)
    sample = val_data[0]
    x = np.expand_dims(sample["image"], axis=0)
    probs = model.predict(x, verbose=0)
    require(probs.shape == (1, 288, 512, 1), f"prediction shape {probs.shape}", issues)
    require(float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0, "probabilities in [0,1]", issues)
    require(float(probs.std()) > 1e-6, "predictions are non-constant", issues)

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        return model.predict(batch, verbose=0)

    # Guard: test must raise
    try:
        evaluate_predictor_on_split(predict_fn, "test", threshold=VAL_THR, allow_test=False)
        require(False, "test evaluation is blocked by default", issues)
    except RuntimeError as exc:
        require("Test-split evaluation is blocked" in str(exc), "test evaluation raises RuntimeError", issues)

    # Spot-check val metrics at frozen threshold (tolerance for float/backend noise)
    val = evaluate_predictor_on_split(predict_fn, "val", threshold=VAL_THR, allow_test=False)
    frozen_dice = float(e1["validation_metrics"]["mean_per_image"]["mean_dice"])
    reload_dice = float(val["mean_per_image"]["mean_dice"])
    require(
        abs(reload_dice - frozen_dice) < 1e-4,
        f"reloaded val mean Dice ≈ frozen ({reload_dice:.6f} vs {frozen_dice:.6f})",
        issues,
    )

    # --- Non-canonical run: metadata required; checkpoint is historical-only ---
    print("\n[4] Non-canonical E1_1501 (historical, non-blocking if weights missing)")
    abandoned = root / "experiments" / "metadata" / "E1_cnn_scratch_20261108_1501.json"
    abandoned_ckpt = root / "checkpoints" / "E1_cnn_scratch_20261108_1501" / "best.keras"
    require(abandoned.is_file(), "abandoned E1_1501 metadata preserved", issues)
    historical_notes: list[str] = []
    if abandoned_ckpt.is_file():
        print("  OK  abandoned E1_1501 checkpoint preserved")
        abandoned_ckpt_status = "present"
    else:
        note = (
            "abandoned E1_1501 checkpoint unavailable (disk cleanup); "
            "metadata retained; not required for canonical reproducibility"
        )
        print(f"NOTE {note}")
        historical_notes.append(note)
        abandoned_ckpt_status = "unavailable"

    qual = root / "experiments" / "figures" / "phase3_E1_val_qualitative.png"
    qual_sidecar = root / "experiments" / "figures" / "phase3_E1_val_qualitative.REGENERATED.json"
    qualitative_provenance = None
    if qual_sidecar.is_file():
        qualitative_provenance = "regenerated_documented"
        print("  INFO qualitative figure has REGENERATED sidecar provenance")
    elif qual.is_file():
        qualitative_provenance = "original_or_undocumented"
    else:
        qualitative_provenance = "missing"

    # Write audit report sidecar (does not alter frozen metrics)
    report = {
        "audit": "phase3_freeze",
        "canonical_e0": CANONICAL_E0,
        "canonical_e1": CANONICAL_E1,
        "abandoned_e1": "E1_cnn_scratch_20261108_1501",
        "abandoned_e1_checkpoint_status": abandoned_ckpt_status,
        "historical_notes": historical_notes,
        "qualitative_figure_provenance": qualitative_provenance,
        "split_manifest_sha256": split_hash,
        "checkpoint_sha256": ckpt_hash,
        "training_threshold": TRAIN_THR,
        "selected_validation_threshold": VAL_THR,
        "reloaded_val_mean_dice": reload_dice,
        "frozen_val_mean_dice": frozen_dice,
        "tensorflow": tf.__version__,
        "issues": issues,
        "status": "pass" if not issues else "fail",
    }
    out = root / "experiments" / "metadata" / "phase3_freeze_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out.relative_to(root)}")
    print(f"STATUS: {report['status'].upper()} ({len(issues)} issue(s))")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
