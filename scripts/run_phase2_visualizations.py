"""Generate Phase-2 preprocessing figures (512x288 pipeline). Does not train models."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from augmentation import AugmentationConfig, augment_pair  # noqa: E402
from dataset import RoadAnomalyDataset, load_split_manifest  # noqa: E402
def _overlay(image: np.ndarray, mask: np.ndarray, color=(220, 30, 45), alpha=0.45) -> np.ndarray:
    painted = image.copy()
    region = mask > 0
    if not region.any():
        return painted
    tint = np.array(color, dtype=np.float32)
    pixels = painted[region].astype(np.float32)
    painted[region] = np.clip(pixels * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
    return painted


def save_size_comparison(figure_dir: Path) -> Path:
    data = RoadAnomalyDataset("train", normalize="scale_0_1", augment=False)
    # Pick a mid-sized animal scene if present.
    frame_ids = [r.frame_id for r in data.records]
    preferred = "animals17_E6_near_Kvaenangen"
    index = frame_ids.index(preferred) if preferred in frame_ids else 0
    sample = data[index]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].imshow(sample["image_uint8"])
    axes[0].set_title("Prepared RGB 512×288")
    axes[1].imshow(sample["mask_uint8"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Binary mask {0,1}")
    axes[2].imshow(_overlay(sample["image_uint8"], sample["mask_uint8"]))
    axes[2].set_title("Overlay (no letterbox bars)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"Phase 2 preprocessing — {sample['frame_id']} — shape {sample['image_uint8'].shape[1]}×{sample['image_uint8'].shape[0]}",
        fontsize=12,
    )
    fig.tight_layout()
    out = figure_dir / "phase2_01_512x288_example.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def save_augmentation_examples(figure_dir: Path) -> Path:
    data = RoadAnomalyDataset("train", normalize="scale_0_1", augment=False)
    sample = data[0]
    cfg = AugmentationConfig.from_project_config()
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.2))
    axes[0, 0].imshow(sample["image_uint8"])
    axes[0, 0].set_title("Original prepared")
    axes[1, 0].imshow(_overlay(sample["image_uint8"], sample["mask_uint8"]))
    axes[1, 0].set_title("Mask overlay")
    for col, seed in enumerate([1, 2, 3], start=1):
        image_a, mask_a = augment_pair(
            sample["image_uint8"],
            sample["mask_uint8"],
            rng=np.random.default_rng(seed),
            config=cfg,
        )
        axes[0, col].imshow(image_a)
        axes[0, col].set_title(f"Augment seed={seed}")
        axes[1, col].imshow(_overlay(image_a, mask_a))
        axes[1, col].set_title(f"Synced mask seed={seed}")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Train-only augmentation keeps image and mask aligned", fontsize=12)
    fig.tight_layout()
    out = figure_dir / "phase2_02_augmentation_sync.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def save_split_grid(figure_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.5))
    for row, split in enumerate(("train", "val", "test")):
        data = RoadAnomalyDataset(split, normalize="scale_0_1", augment=False)
        for col in range(3):
            sample = data[col]
            axes[row, col].imshow(_overlay(sample["image_uint8"], sample["mask_uint8"]))
            axes[row, col].set_title(f"{split}: {sample['category']}", fontsize=9)
            axes[row, col].axis("off")
    fig.suptitle("Same 512×288 pipeline on train / val / test (no augmentation on val/test)", fontsize=12)
    fig.tight_layout()
    out = figure_dir / "phase2_03_split_examples.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def save_summary_table(figure_dir: Path) -> Path:
    manifest = load_split_manifest()
    rows = []
    for split in ("train", "val", "test"):
        data = RoadAnomalyDataset(split, augment=False)
        anom = [data[i]["anomaly_pixels"] for i in range(len(data))]
        rows.append(
            {
                "split": split,
                "n": len(data),
                "shape": "288x512",
                "min_anom_px": int(min(anom)),
                "median_anom_px": int(np.median(anom)),
                "max_anom_px": int(max(anom)),
            }
        )
    # Render as a simple text figure for the report pack.
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.axis("off")
    lines = ["Phase 2 pipeline summary (from original frames → 512×288, labels 0/1)", ""]
    for row in rows:
        lines.append(
            f"{row['split']:5}  n={row['n']:2}  shape={row['shape']}  "
            f"anomaly px min/med/max = {row['min_anom_px']}/{row['median_anom_px']}/{row['max_anom_px']}"
        )
    lines.append("")
    lines.append(f"Manifest rows locked: {len(manifest)} | letterbox training input: disabled")
    ax.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
    out = figure_dir / "phase2_04_pipeline_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    figure_dir = ROOT / "experiments" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        save_size_comparison(figure_dir),
        save_augmentation_examples(figure_dir),
        save_split_grid(figure_dir),
        save_summary_table(figure_dir),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
