"""Dataset loading from the frozen 42/9/9 split.

Loads original WebP frames and official semantic masks, then applies the
Phase-1 preprocessing (512x288, labels 0/1). Does not call shutil.rmtree and
does not rewrite data/processed letterbox artefacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd

from augmentation import AugmentationConfig, augment_pair
from config import assert_no_letterbox, load_config, project_root, resolve_path
from preprocessing import (
    NormalizeMode,
    load_rgb_image,
    load_semantic_mask,
    normalize_image,
    preprocess_pair,
)

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SampleRecord:
    frame_id: str
    category: str
    scene_group: str
    split: SplitName
    image_path: Path
    mask_path: Path
    anomaly_ratio_original: float


def assignment_sha256_from_manifest(manifest: pd.DataFrame) -> str:
    pairs = sorted(
        (str(row.frame_id), str(row.split)) for row in manifest.itertuples(index=False)
    )
    return hashlib.sha256(repr(pairs).encode("utf-8")).hexdigest()


def verify_split_lock(manifest: pd.DataFrame, config: dict | None = None) -> str:
    cfg = config or load_config()
    expected = cfg["split"]["assignment_sha256"]
    actual = assignment_sha256_from_manifest(manifest)
    if actual != expected:
        raise ValueError(
            "Split assignment hash mismatch. Refusing to load a shuffled split.\n"
            f"expected: {expected}\nactual:   {actual}"
        )
    lock_file = project_root() / cfg["split"]["lock_file"]
    if lock_file.exists():
        payload = json.loads(lock_file.read_text(encoding="utf-8"))
        if payload.get("assignment_sha256") != actual:
            raise ValueError("split_lock.json does not match the manifest assignment hash")
    return actual


def original_paths(frame_id: str, frames_dir: Path) -> tuple[Path, Path]:
    image_path = frames_dir / f"{frame_id}.webp"
    mask_path = frames_dir / f"{frame_id}.labels" / "labels_semantic.png"
    return image_path, mask_path


def load_split_manifest(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    assert_no_letterbox(cfg)
    path = resolve_path(cfg, "split_manifest")
    manifest = pd.read_csv(path)
    required = {"frame_id", "category", "scene_group", "split", "anomaly_ratio_original"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    verify_split_lock(manifest, cfg)

    counts = manifest["split"].value_counts().to_dict()
    expected = {
        "train": cfg["split"]["train"],
        "val": cfg["split"]["val"],
        "test": cfg["split"]["test"],
    }
    for split, n_expected in expected.items():
        if int(counts.get(split, 0)) != int(n_expected):
            raise ValueError(f"Split {split}: expected {n_expected}, found {counts.get(split, 0)}")

    # Leakage check: scene groups must not cross splits.
    crossing = int(manifest.groupby("scene_group")["split"].nunique().gt(1).sum())
    if crossing:
        raise ValueError(f"{crossing} scene groups appear in more than one split")
    return manifest.sort_values(["split", "category", "frame_id"]).reset_index(drop=True)


def iter_records(
    split: SplitName | None = None,
    config: dict | None = None,
) -> list[SampleRecord]:
    cfg = config or load_config()
    manifest = load_split_manifest(cfg)
    frames_dir = resolve_path(cfg, "frames_dir")
    if split is not None:
        manifest = manifest.loc[manifest["split"] == split]
    records: list[SampleRecord] = []
    for row in manifest.itertuples(index=False):
        image_path, mask_path = original_paths(str(row.frame_id), frames_dir)
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        records.append(
            SampleRecord(
                frame_id=str(row.frame_id),
                category=str(row.category),
                scene_group=str(row.scene_group),
                split=str(row.split),  # type: ignore[arg-type]
                image_path=image_path,
                mask_path=mask_path,
                anomaly_ratio_original=float(row.anomaly_ratio_original),
            )
        )
    return records


def load_processed_sample(
    record: SampleRecord,
    *,
    normalize: NormalizeMode = "scale_0_1",
    augment: bool = False,
    rng: np.random.Generator | None = None,
    aug_config: AugmentationConfig | None = None,
) -> dict:
    image = load_rgb_image(record.image_path)
    mask = load_semantic_mask(record.mask_path)
    packed = preprocess_pair(
        image,
        mask,
        normalize="none",  # normalize after optional aug on uint8 space
        frame_id=record.frame_id,
    )
    image_u8 = packed["image_uint8"]
    mask_bin = packed["mask"]

    if augment:
        if record.split != "train":
            raise ValueError("Augmentation is allowed only on the train split")
        image_u8, mask_bin = augment_pair(image_u8, mask_bin, rng=rng, config=aug_config)

    image_n = normalize_image(image_u8, normalize)
    return {
        "frame_id": record.frame_id,
        "category": record.category,
        "scene_group": record.scene_group,
        "split": record.split,
        "image": image_n.astype(np.float32),
        "mask": mask_bin.astype(np.float32)[..., None],  # (H, W, 1) for Keras
        "image_uint8": image_u8,
        "mask_uint8": mask_bin,
        "anomaly_pixels": int(mask_bin.sum()),
        "anomaly_ratio": float(mask_bin.mean()),
    }


class RoadAnomalyDataset:
    """Simple indexed dataset for NumPy / Keras Sequence usage."""

    def __init__(
        self,
        split: SplitName,
        *,
        normalize: NormalizeMode = "scale_0_1",
        augment: bool = False,
        seed: int = 42,
        config: dict | None = None,
    ) -> None:
        self.split = split
        self.normalize = normalize
        self.augment = augment and split == "train"
        self.seed = seed
        self.config = config or load_config()
        self.records = iter_records(split, self.config)
        self.aug_config = AugmentationConfig.from_project_config(self.config)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng(self.seed + index * 1_000_003) if self.augment else None
        return load_processed_sample(
            self.records[index],
            normalize=self.normalize,
            augment=self.augment,
            rng=rng,
            aug_config=self.aug_config,
        )

    def iter_batches(self, batch_size: int, shuffle: bool = False) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(len(self))
        if shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            images = []
            masks = []
            for index in batch_idx:
                sample = self[int(index)]
                images.append(sample["image"])
                masks.append(sample["mask"])
            yield np.stack(images, axis=0), np.stack(masks, axis=0)


def summarize_split_shapes(normalize: NormalizeMode = "scale_0_1") -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        data = RoadAnomalyDataset(split, normalize=normalize, augment=False)
        sample = data[0]
        rows.append(
            {
                "split": split,
                "n": len(data),
                "image_shape": tuple(sample["image"].shape),
                "mask_shape": tuple(sample["mask"].shape),
                "mask_values": sorted(np.unique(sample["mask_uint8"]).tolist()),
                "min_anomaly_pixels": min(data[i]["anomaly_pixels"] for i in range(len(data))),
            }
        )
    return pd.DataFrame(rows)
