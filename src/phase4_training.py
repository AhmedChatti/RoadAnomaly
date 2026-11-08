"""Phase 4 training utilities (E2a / E2b).

Training and ImageNet downloads are gated. Calling train_* without
authorization flags raises RuntimeError so accidental execution is safe.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import load_config, project_root, resolve_path
from reproducibility import (
    build_experiment_metadata_base,
    git_commit_sha,
    sha256_file,
    write_json,
)

PHASE4_CONFIG_PATH = project_root() / "configs" / "phase4.json"


def load_phase4_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else PHASE4_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def phase4_artefact_dirs(p4: dict[str, Any] | None = None) -> dict[str, Path]:
    """Resolve Phase 4 output directories (isolated under …/phase4/)."""
    p4 = p4 or load_phase4_config()
    cfg = load_config()
    root = project_root()
    return {
        "checkpoints": resolve_path(cfg, "checkpoints_dir") / "phase4",
        "metadata": resolve_path(cfg, "experiments_metadata_dir") / "phase4",
        "results": resolve_path(cfg, "experiments_results_dir") / "phase4",
        "figures": resolve_path(cfg, "experiments_figures_dir") / "phase4",
        "repo_root": root,
        "never_write_to": [root / rel for rel in p4["artefacts"]["never_write_to"]],
    }


def assert_not_writing_to_phase3(path: Path, p4: dict[str, Any] | None = None) -> None:
    dirs = phase4_artefact_dirs(p4)
    resolved = path.resolve()
    for blocked in dirs["never_write_to"]:
        blocked_resolved = blocked.resolve()
        if blocked_resolved.is_file():
            if resolved == blocked_resolved:
                raise RuntimeError(f"Refusing to overwrite frozen Phase 3 artefact: {blocked}")
            continue
        try:
            resolved.relative_to(blocked_resolved)
            raise RuntimeError(f"Refusing to write into frozen Phase 3 path: {blocked}")
        except ValueError:
            continue
    phase4_roots = (dirs["checkpoints"], dirs["metadata"], dirs["results"], dirs["figures"])
    for root in phase4_roots:
        try:
            resolved.relative_to(root.resolve())
            return
        except ValueError:
            continue
    raise RuntimeError(
        f"Phase 4 artefact path must be under checkpoints/phase4 or experiments/*/phase4; got {resolved}"
    )


def assert_required_git_ancestry(
    p4: dict[str, Any] | None = None,
    *,
    repo_dir: Path | None = None,
) -> dict[str, str]:
    """Fail fast unless configs.phase4.git_commit_required is an ancestor of HEAD.

    Does not require HEAD to equal the required commit exactly.
    Does not silently skip when Git is unavailable.
    """
    p4 = p4 or load_phase4_config()
    required = str(p4["git_commit_required"]).strip()
    root = repo_dir or project_root()

    def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                args,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise RuntimeError(
                "Git is required for Phase 4 training/ImageNet authorization, "
                f"but could not be executed: {exc}"
            ) from exc

    probe = _run(["git", "rev-parse", "--is-inside-work-tree"])
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        raise RuntimeError(
            "Git repository unavailable; cannot validate phase4.git_commit_required="
            f"{required}. Phase 4 training/ImageNet download is blocked."
        )

    exists = _run(["git", "cat-file", "-e", f"{required}^{{commit}}"])
    if exists.returncode != 0:
        raise RuntimeError(
            f"Required infrastructure commit {required} was not found in this repository. "
            "Fetch/pull the approved RoadAnomaly history before Phase 4 training."
        )

    head = _run(["git", "rev-parse", "HEAD"])
    if head.returncode != 0 or not head.stdout.strip():
        raise RuntimeError("Unable to resolve git HEAD for Phase 4 ancestry check.")
    head_sha = head.stdout.strip()

    ancestor = _run(["git", "merge-base", "--is-ancestor", required, "HEAD"])
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"Required infrastructure commit {required} is not an ancestor of HEAD ({head_sha}). "
            "Phase 4 refuses unrelated histories. Checkout/rebase onto a descendant of the "
            "approved infrastructure commit before training."
        )
    return {"required": required, "head": head_sha, "ancestry": "ok"}


def _require_training_authorization(
    *,
    authorize_training: bool,
    authorize_imagenet_download: bool,
    weights: str | None,
) -> None:
    if authorize_training != authorize_imagenet_download:
        raise RuntimeError(
            "Both authorize_training and authorize_imagenet_download must be supplied together "
            "(got "
            f"authorize_training={authorize_training}, "
            f"authorize_imagenet_download={authorize_imagenet_download})."
        )
    if not authorize_training:
        raise RuntimeError(
            "Phase 4 training is gated. Pass authorize_training=True only after "
            "explicit user approval for a Colab training run."
        )
    if weights == "imagenet" and not authorize_imagenet_download:
        raise RuntimeError(
            "ImageNet weight download is gated. Pass authorize_imagenet_download=True "
            "only after explicit user approval."
        )
    # Ancestry check before any download/training work.
    assert_required_git_ancestry()


def build_phase4_model_for_experiment(
    experiment: str,
    p4: dict[str, Any] | None = None,
    *,
    weights: str | None = None,
    authorize_imagenet_download: bool = False,
):
    """Construct E2a/E2b architecture. Defaults to weights=None (no download)."""
    from models.unet_mobilenetv2 import build_unet_mobilenetv2, configure_e2b_finetuning

    p4 = p4 or load_phase4_config()
    if weights == "imagenet" and not authorize_imagenet_download:
        raise RuntimeError(
            "Refusing to build with weights='imagenet' without authorize_imagenet_download=True"
        )
    if weights not in (None, "imagenet"):
        raise ValueError(weights)

    height = int(p4["input"]["height"])
    width = int(p4["input"]["width"])
    dropout = float(p4["decoder"]["dropout"])
    filters = tuple(int(x) for x in p4["decoder"]["filters"][:4])
    final_conv_filters = int(p4["decoder"].get("final_conv_filters", 16))
    skips = tuple(p4["encoder"]["skip_layer_names"])
    bottleneck = p4["encoder"]["bottleneck_layer_name"]

    common = dict(
        input_height=height,
        input_width=width,
        weights=weights,  # type: ignore[arg-type]
        decoder_filters=filters,
        final_conv_filters=final_conv_filters,
        dropout=dropout,
        skip_layer_names=skips,
        bottleneck_layer_name=bottleneck,
    )

    if experiment == "E2a":
        model = build_unet_mobilenetv2(
            encoder_trainable=False,
            name="E2a_unet_mobilenetv2",
            **common,
        )
        return model, {"experiment": "E2a", "finetune": None}

    if experiment == "E2b":
        model = build_unet_mobilenetv2(
            encoder_trainable=True,
            name="E2b_unet_mobilenetv2",
            **common,
        )
        finetune = configure_e2b_finetuning(
            model,
            finetune_from_layer=p4["E2b"]["finetune_from_layer"],
        )
        return model, {"experiment": "E2b", "finetune": finetune}

    raise ValueError(f"Unknown Phase 4 experiment: {experiment}")


def make_phase4_metadata(
    *,
    experiment_id: str,
    experiment: str,
    model_name: str,
    architecture: str,
    hyperparameters: dict[str, Any],
    p4: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p4 = p4 or load_phase4_config()
    meta = build_experiment_metadata_base(
        experiment_id=experiment_id,
        model_name=model_name,
        architecture=architecture,
        hyperparameters=hyperparameters,
    )
    meta["phase"] = 4
    meta["normalize"] = p4["input"]["normalize"]
    meta["threshold_selection"]["default_during_training"] = p4["training_threshold"]
    meta["git_commit_sha"] = git_commit_sha()
    meta["status"] = "initialized_no_training"
    return meta


def threshold_grid_from_config(p4: dict[str, Any] | None = None) -> list[float]:
    p4 = p4 or load_phase4_config()
    thr = p4["threshold_selection"]
    start = float(thr["grid_start"])
    stop = float(thr["grid_stop"])
    steps = int(thr["grid_steps"])
    if steps < 2:
        raise ValueError("threshold_selection.grid_steps must be >= 2")
    import numpy as np

    return [round(float(x), 2) for x in np.linspace(start, stop, steps)]


def finalize_phase4_threshold_selection(
    *,
    model,
    experiment_id: str,
    metadata: dict[str, Any],
    p4: dict[str, Any] | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run validation-only threshold selection on the restored best checkpoint model."""
    from phase4_evaluation import select_phase4_threshold_on_validation

    p4 = p4 or load_phase4_config()
    dirs = phase4_artefact_dirs(p4)
    dirs["results"].mkdir(parents=True, exist_ok=True)
    dirs["metadata"].mkdir(parents=True, exist_ok=True)

    def predict_fn(batch):
        return model.predict(batch, verbose=0)

    try:
        thr_info = select_phase4_threshold_on_validation(
            predict_fn,
            batch_size=int(batch_size if batch_size is not None else p4["batch_size"]),
            thresholds=threshold_grid_from_config(p4),
        )
    except Exception as exc:  # noqa: BLE001
        metadata["status"] = "incomplete_threshold_selection_failed"
        metadata["threshold_selection"] = {
            "split": "val",
            "metric": "mean_dice",
            "value": None,
            "default_during_training": float(p4["training_threshold"]),
            "error": str(exc),
        }
        save_phase4_metadata(dirs["metadata"] / f"{experiment_id}.json", metadata)
        raise RuntimeError(
            f"Validation threshold selection failed for {experiment_id}; "
            "experiment marked incomplete. No fabricated threshold was written."
        ) from exc

    grid_path = dirs["results"] / f"{experiment_id}_val_threshold_grid.csv"
    thr_info["grid"].to_csv(grid_path, index=False)
    assert_not_writing_to_phase3(grid_path, p4)

    best_thr = float(thr_info["best_threshold"])
    metadata["threshold_selection"] = {
        "split": "val",
        "metric": "mean_dice",
        "value": best_thr,
        "default_during_training": float(p4["training_threshold"]),
        "grid_best_val_mean_dice": float(thr_info["best_val_mean_dice"]),
        "grid_csv": str(grid_path).replace("\\", "/"),
        "notes": (
            "value is the operational threshold selected on validation mean Dice. "
            "default_during_training is the fixed threshold used by MeanDiceCallback "
            "for checkpoint selection."
        ),
    }
    metadata["validation_metrics"] = {
        "mean_per_image": thr_info["best_result"]["mean_per_image"],
        "global": thr_info["best_result"]["global"],
        "threshold": best_thr,
        "split": "val",
    }
    if metadata.get("status") == "completed":
        metadata["status"] = "completed_with_val_threshold"
    return metadata


def _fit_phase4_model(
    *,
    experiment: str,
    experiment_id: str,
    model,
    learning_rate: float,
    p4: dict[str, Any],
) -> dict[str, Any]:
    """Shared train/val fit loop. Caller must already have passed authorization gates."""
    import time

    import tensorflow as tf
    from tensorflow import keras

    from losses import bce_dice_loss
    from training import MeanDiceCallback, RoadAnomalySequence, set_global_seed

    seed = int(p4["seed"])
    set_global_seed(seed)
    batch_size = int(p4["batch_size"])
    epochs = int(p4["epochs_max"])
    patience = int(p4["early_stopping_patience"])
    normalize = p4["input"]["normalize"]
    positive_weight = float(p4["loss"]["positive_weight"])

    dirs = phase4_artefact_dirs(p4)
    for key in ("checkpoints", "metadata", "results", "figures"):
        dirs[key].mkdir(parents=True, exist_ok=True)

    ckpt_dir = dirs["checkpoints"] / experiment_id
    ckpt_path = ckpt_dir / "best.keras"
    assert_not_writing_to_phase3(ckpt_path, p4)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_seq = RoadAnomalySequence(
        "train",
        batch_size=batch_size,
        normalize=normalize,  # type: ignore[arg-type]
        augment=True,
        seed=seed,
        shuffle=True,
    )
    val_seq = RoadAnomalySequence(
        "val",
        batch_size=batch_size,
        normalize=normalize,  # type: ignore[arg-type]
        augment=False,
        seed=seed,
        shuffle=False,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=bce_dice_loss(
            positive_weight=positive_weight,
            dice_weight=float(p4["loss"]["dice_weight"]),
        ),
        metrics=[keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )
    dice_cb = MeanDiceCallback(
        val_seq,
        ckpt_path,
        patience=patience,
        threshold=float(p4["training_threshold"]),
    )

    started = time.time()
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=[dice_cb],
        verbose=1,
    )
    duration = time.time() - started

    best_model = keras.models.load_model(ckpt_path, compile=False)
    ckpt_sha = sha256_file(ckpt_path)

    meta = make_phase4_metadata(
        experiment_id=experiment_id,
        experiment=experiment,
        model_name="unet_mobilenetv2",
        architecture=f"U-Net + MobileNetV2 ({experiment})",
        hyperparameters={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs_max": epochs,
            "positive_weight": positive_weight,
            "normalize": normalize,
        },
        p4=p4,
    )
    meta.update(
        {
            "status": "completed",
            "epochs_run": len(history.history.get("loss", [])),
            "best_epoch": dice_cb.best_epoch,
            "best_val_metric_name": "val_mean_dice",
            "best_val_metric_value": dice_cb.best,
            "checkpoint_path": str(ckpt_path).replace("\\", "/"),
            "checkpoint_sha256": ckpt_sha,
            "duration_seconds": duration,
            "hardware": "gpu" if tf.config.list_physical_devices("GPU") else "cpu",
            "history": {
                "loss": [float(x) for x in history.history.get("loss", [])],
                "val_loss": [float(x) for x in history.history.get("val_loss", [])],
                "binary_accuracy": [float(x) for x in history.history.get("binary_accuracy", [])],
                "val_binary_accuracy": [
                    float(x) for x in history.history.get("val_binary_accuracy", [])
                ],
                "val_mean_dice": dice_cb.history,
            },
            "threshold_selection": {
                "split": "val",
                "metric": "mean_dice",
                "value": None,
                "default_during_training": float(p4["training_threshold"]),
            },
            "artefact_locations": {
                "metadata": str(dirs["metadata"]).replace("\\", "/"),
                "results": str(dirs["results"]).replace("\\", "/"),
                "figures": str(dirs["figures"]).replace("\\", "/"),
                "checkpoint": str(ckpt_path).replace("\\", "/"),
            },
        }
    )

    # Mandatory validation-only operational threshold selection on best checkpoint.
    meta = finalize_phase4_threshold_selection(
        model=best_model,
        experiment_id=experiment_id,
        metadata=meta,
        p4=p4,
        batch_size=batch_size,
    )
    try:
        fig_paths = save_phase4_training_figures(
            experiment_id=experiment_id,
            metadata=meta,
            model=best_model,
            p4=p4,
        )
        meta["figure_paths"] = {k: str(v).replace("\\", "/") for k, v in fig_paths.items()}
    except Exception as exc:  # noqa: BLE001
        meta["figure_export_error"] = str(exc)
    meta_path = dirs["metadata"] / f"{experiment_id}.json"
    save_phase4_metadata(meta_path, meta)
    return {
        "model": best_model,
        "metadata": meta,
        "metadata_path": meta_path,
        "checkpoint_path": ckpt_path,
    }


def save_phase4_training_figures(
    *,
    experiment_id: str,
    metadata: dict[str, Any],
    model,
    p4: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write training curves + a small validation qualitative panel (val only)."""
    import matplotlib.pyplot as plt
    import numpy as np

    from dataset import RoadAnomalyDataset
    from phase4_evaluation import evaluate_phase4_on_split

    p4 = p4 or load_phase4_config()
    dirs = phase4_artefact_dirs(p4)
    dirs["figures"].mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    hist = metadata.get("history") or {}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    loss = hist.get("loss") or []
    val_loss = hist.get("val_loss") or []
    val_dice = hist.get("val_mean_dice") or []
    epochs = np.arange(1, len(loss) + 1)
    if loss:
        axes[0].plot(epochs, loss, label="train loss")
    if val_loss:
        axes[0].plot(np.arange(1, len(val_loss) + 1), val_loss, label="val loss")
    axes[0].set_title("Loss")
    axes[0].legend(frameon=False)
    axes[0].set_xlabel("Epoch")
    if val_dice:
        axes[1].plot(np.arange(1, len(val_dice) + 1), val_dice, color="#54A24B")
        best_epoch = metadata.get("best_epoch")
        if best_epoch is not None:
            axes[1].axvline(best_epoch, color="#E45756", linestyle="--", label=f"best @{best_epoch}")
        axes[1].legend(frameon=False)
    axes[1].set_title("Validation mean Dice / image @ train thr")
    axes[1].set_xlabel("Epoch")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"{experiment_id} training curves")
    fig.tight_layout()
    curves = dirs["figures"] / f"{experiment_id}_training_curves.png"
    assert_not_writing_to_phase3(curves, p4)
    fig.savefig(curves, dpi=140, facecolor="white")
    plt.close(fig)
    out["training_curves"] = curves

    thr = float((metadata.get("threshold_selection") or {}).get("value") or p4["training_threshold"])

    def predict_fn(batch):
        return model.predict(batch, verbose=0)

    val = evaluate_phase4_on_split(predict_fn, "val", threshold=thr, batch_size=int(p4["batch_size"]))
    data = RoadAnomalyDataset("val", normalize=p4["input"]["normalize"], augment=False)
    table = val["per_image"].sort_values("dice")
    picks = [
        ("worst Dice", table.iloc[0]["frame_id"]),
        ("best Dice", table.iloc[-1]["frame_id"]),
        ("smallest anomaly", table.sort_values("anomaly_ratio").iloc[0]["frame_id"]),
    ]
    id_to_index = {rec.frame_id: i for i, rec in enumerate(data.records)}
    fig, axes = plt.subplots(len(picks), 3, figsize=(10.5, 8.5))
    for row, (title, frame_id) in enumerate(picks):
        sample = data[id_to_index[frame_id]]
        prob = model.predict(sample["image"][None, ...], verbose=0)[0, ..., 0]
        pred = (prob >= thr).astype(np.uint8)
        axes[row, 0].imshow(sample["image_uint8"])
        axes[row, 0].set_ylabel(title, fontsize=9)
        axes[row, 0].set_title(frame_id, fontsize=8)
        axes[row, 1].imshow(sample["mask_uint8"], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("GT" if row == 0 else "")
        axes[row, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Pred" if row == 0 else "")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{experiment_id} val qualitative (thr={thr:.2f})", fontsize=11)
    fig.tight_layout()
    qual = dirs["figures"] / f"{experiment_id}_val_qualitative.png"
    assert_not_writing_to_phase3(qual, p4)
    fig.savefig(qual, dpi=130, facecolor="white")
    plt.close(fig)
    out["val_qualitative"] = qual
    return out


def train_e2a(
    *,
    authorize_training: bool = False,
    authorize_imagenet_download: bool = False,
    weights: str | None = "imagenet",
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """E2a training entrypoint — blocked unless both authorizations are granted."""
    _require_training_authorization(
        authorize_training=authorize_training,
        authorize_imagenet_download=authorize_imagenet_download,
        weights=weights,
    )
    p4 = load_phase4_config()
    exp_id = experiment_id or (
        f"{p4['E2a']['experiment_prefix']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
    )
    model, info = build_phase4_model_for_experiment(
        "E2a",
        p4,
        weights=weights,
        authorize_imagenet_download=authorize_imagenet_download,
    )
    result = _fit_phase4_model(
        experiment="E2a",
        experiment_id=exp_id,
        model=model,
        learning_rate=float(p4["E2a"]["learning_rate"]),
        p4=p4,
    )
    result["metadata"]["encoder_config"] = info
    save_phase4_metadata(result["metadata_path"], result["metadata"])
    return result


def train_e2b(
    *,
    e2a_checkpoint: Path | str,
    authorize_training: bool = False,
    authorize_imagenet_download: bool = False,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """E2b training entrypoint — blocked unless authorized; loads best E2a weights."""
    from tensorflow import keras

    from models.unet_mobilenetv2 import ENCODER_NAMES_ATTR, configure_e2b_finetuning, freeze_batch_norm

    _require_training_authorization(
        authorize_training=authorize_training,
        authorize_imagenet_download=authorize_imagenet_download,
        weights="imagenet",
    )
    p4 = load_phase4_config()
    ckpt = Path(e2a_checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"E2a checkpoint not found: {ckpt}")

    model, _ = build_phase4_model_for_experiment(
        "E2b",
        p4,
        weights=None,
        authorize_imagenet_download=False,
    )
    loaded = keras.models.load_model(ckpt, compile=False)
    model.set_weights(loaded.get_weights())
    finetune = configure_e2b_finetuning(
        model, finetune_from_layer=p4["E2b"]["finetune_from_layer"]
    )
    freeze_batch_norm(model)

    exp_id = experiment_id or (
        f"{p4['E2b']['experiment_prefix']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
    )
    result = _fit_phase4_model(
        experiment="E2b",
        experiment_id=exp_id,
        model=model,
        learning_rate=float(p4["E2b"]["learning_rate"]),
        p4=p4,
    )
    result["metadata"]["initialized_from"] = str(ckpt).replace("\\", "/")
    result["metadata"]["finetune"] = finetune
    result["metadata"][ENCODER_NAMES_ATTR] = sorted(getattr(model, ENCODER_NAMES_ATTR, set()))
    save_phase4_metadata(result["metadata_path"], result["metadata"])
    return result


def record_checkpoint_sha(path: Path) -> str:
    return sha256_file(path)


def save_phase4_metadata(path: Path, payload: dict[str, Any]) -> Path:
    assert_not_writing_to_phase3(path)
    return write_json(path, payload)
