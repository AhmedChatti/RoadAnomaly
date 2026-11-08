"""Training helpers for Phase 3+ experiments."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import load_config, project_root
from dataset import RoadAnomalyDataset, assignment_sha256_from_manifest, load_split_manifest
from losses import bce_dice_loss
from preprocessing import NormalizeMode


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


class RoadAnomalySequence(keras.utils.PyDataset):
    def __init__(
        self,
        split: str,
        *,
        batch_size: int,
        normalize: NormalizeMode = "scale_0_1",
        augment: bool = False,
        seed: int = 42,
        shuffle: bool = False,
    ) -> None:
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
        self.dataset = RoadAnomalyDataset(
            split,  # type: ignore[arg-type]
            normalize=normalize,
            augment=augment,
            seed=seed,
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indices = np.arange(len(self.dataset))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, batch_index: int):
        start = batch_index * self.batch_size
        batch_idx = self.indices[start : start + self.batch_size]
        images = []
        masks = []
        for index in batch_idx:
            sample = self.dataset[int(index)]
            images.append(sample["image"])
            masks.append(sample["mask"])
        return np.stack(images, axis=0), np.stack(masks, axis=0)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            rng = np.random.default_rng(self.seed + int(time.time()) % 10_000)
            rng.shuffle(self.indices)


class MeanDiceCallback(keras.callbacks.Callback):
    """Track mean per-image Dice on a validation sequence and save the best weights."""

    def __init__(
        self,
        val_sequence: RoadAnomalySequence,
        checkpoint_path: Path,
        patience: int = 12,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.val_sequence = val_sequence
        self.checkpoint_path = Path(checkpoint_path)
        self.patience = patience
        self.threshold = threshold
        self.best = -1.0
        self.wait = 0
        self.best_epoch = 0
        self.history: list[float] = []

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        from metrics import binary_segmentation_metrics, threshold_probabilities

        logs = logs or {}
        dices = []
        for batch_index in range(len(self.val_sequence)):
            x, y = self.val_sequence[batch_index]
            probs = self.model(x, training=False).numpy()
            preds = threshold_probabilities(probs[..., 0], self.threshold)
            for i in range(x.shape[0]):
                m = binary_segmentation_metrics(y[i, ..., 0], preds[i])
                dices.append(m["dice"])
        mean_dice = float(np.mean(dices)) if dices else 0.0
        self.history.append(mean_dice)
        logs["val_mean_dice"] = mean_dice
        print(f" — val_mean_dice={mean_dice:.4f}", flush=True)

        if mean_dice > self.best:
            self.best = mean_dice
            self.best_epoch = epoch + 1
            self.wait = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.checkpoint_path)
            print(f"   saved best checkpoint → {self.checkpoint_path}", flush=True)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(
                    f"   early stopping at epoch {epoch + 1} "
                    f"(best val_mean_dice={self.best:.4f} @ epoch {self.best_epoch})",
                    flush=True,
                )


def software_versions() -> dict[str, str]:
    import PIL

    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "platform": platform.platform(),
    }


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_scratch_fcn(
    *,
    experiment_id: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    positive_weight: float = 15.0,
) -> dict[str, Any]:
    from models.cnn import build_scratch_fcn

    cfg = load_config()
    seed = int(cfg["training"]["seed"])
    set_global_seed(seed)
    epochs = int(epochs if epochs is not None else cfg["training"]["epochs_max"])
    batch_size = int(batch_size if batch_size is not None else cfg["training"]["batch_size"])
    learning_rate = float(
        learning_rate if learning_rate is not None else cfg["training"]["learning_rate_scratch"]
    )
    patience = int(cfg["training"]["early_stopping_patience"])
    height = int(cfg["preprocessing"]["input_height"])
    width = int(cfg["preprocessing"]["input_width"])

    manifest = load_split_manifest(cfg)
    split_hash = assignment_sha256_from_manifest(manifest)

    train_seq = RoadAnomalySequence(
        "train",
        batch_size=batch_size,
        normalize="scale_0_1",
        augment=True,
        seed=seed,
        shuffle=True,
    )
    val_seq = RoadAnomalySequence(
        "val",
        batch_size=batch_size,
        normalize="scale_0_1",
        augment=False,
        seed=seed,
        shuffle=False,
    )

    model = build_scratch_fcn(input_height=height, input_width=width, base_filters=8)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=bce_dice_loss(positive_weight=positive_weight, dice_weight=1.0),
        metrics=[keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )

    root = project_root()
    ckpt_dir = root / "checkpoints" / experiment_id
    ckpt_path = ckpt_dir / "best.keras"
    dice_cb = MeanDiceCallback(val_seq, ckpt_path, patience=patience, threshold=0.5)

    started = time.time()
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=[dice_cb],
        verbose=1,
    )
    duration = time.time() - started

    # Reload best weights for downstream evaluation.
    best_model = keras.models.load_model(
        ckpt_path,
        custom_objects={"bce_dice": bce_dice_loss(positive_weight=positive_weight)},
        compile=False,
    )

    meta = {
        "experiment_id": experiment_id,
        "model_name": "scratch_fcn_unetlite",
        "architecture": "U-Net-lite FCN from scratch (base_filters=8, no BatchNorm)",
        "input_height": height,
        "input_width": width,
        "base_filters": 8,
        "batch_norm": False,
        "split_manifest": "data/processed/split_manifest.csv",
        "split_manifest_sha256": split_hash,
        "seed": seed,
        "optimizer": "adam",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_planned": epochs,
        "epochs_run": len(history.history.get("loss", [])),
        "loss": "weighted_bce + soft_dice",
        "positive_weight": positive_weight,
        "metrics_monitored": ["binary_accuracy", "val_mean_dice"],
        "augmentation": cfg["augmentation"],
        "checkpoint_selection_criterion": "val_mean_dice",
        "best_epoch": dice_cb.best_epoch,
        "best_val_metric_name": "val_mean_dice",
        "best_val_metric_value": dice_cb.best,
        "threshold_selection": {
            "split": "val",
            "metric": "mean_dice",
            "value": None,
            "default_during_training": 0.5,
        },
        "hardware": "local_cpu" if not tf.config.list_physical_devices("GPU") else "gpu",
        "software_versions": software_versions(),
        "duration_seconds": duration,
        "checkpoint_path": str(ckpt_path.relative_to(root)).replace("\\", "/"),
        "history": {
            "loss": [float(x) for x in history.history.get("loss", [])],
            "val_loss": [float(x) for x in history.history.get("val_loss", [])],
            "binary_accuracy": [float(x) for x in history.history.get("binary_accuracy", [])],
            "val_binary_accuracy": [float(x) for x in history.history.get("val_binary_accuracy", [])],
            "val_mean_dice": dice_cb.history,
        },
        "notes": (
            "Phase 3 E1. Test split was not used for selection. "
            "BatchNorm removed because batch_size=2 on CPU caused train/eval collapse "
            "toward all-background predictions."
        ),
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = root / "experiments" / "metadata" / f"{experiment_id}.json"
    write_metadata(meta_path, meta)
    return {"model": best_model, "metadata": meta, "metadata_path": meta_path, "checkpoint_path": ckpt_path}
