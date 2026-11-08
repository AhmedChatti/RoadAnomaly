"""
RoadAnomaly exploration and preparation.

The official files store each road scene as a WebP image and a semantic mask.
In the mask, 0 is ordinary scene content and 2 is an anomalous obstacle.
This module checks that pairing, summarises the data, and writes a
scene-safe train / validation / test set for later segmentation experiments.
"""

from __future__ import annotations

import json
import shutil
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Official encoding from the dataset readme.
BACKGROUND_LABEL = 0
ANOMALY_LABEL = 2
# Stored masks use 255 so the anomaly is visible when the file is opened.
ANOMALY_PIXEL = 255
CANVAS_SIZE = 512

# Same physical scene filmed more than once. These must not be split across
# train, validation, and test, or a model could be scored on a near-copy of
# an image it has already seen.
DUPLICATE_SCENES = {
    "cardboard_czech_hedgehog": [
        "obstacles03_cardboard_czech_hedgehog",
        "obstacles04_cardboard_czech_hedgehog",
        "obstacles05_cardboard_czech_hedgehog",
        "obstacles06_cardboard_czech_hedgehog",
    ],
    "rocks_sequence": [
        "obstacles12_rocks2",
        "obstacles12_rocks4",
        "obstacles12_rocks5",
        "obstacles12_rocks6",
    ],
    "lost_tires_sequence": [
        "lost_tires03",
        "lost_tires08",
        "lost_tires09",
    ],
}

# Independent images only. Duplicate scenes are kept entirely in training.
# Counts are chosen so the hold-out sets stay close to 15% each (9 and 9)
# while every category that can be split is present in each subset.
HOLDOUT_QUOTAS = {
    "animals": {"val": 5, "test": 5},
    "obstacles": {"val": 1, "test": 1},
    "cones": {"val": 2, "test": 2},
    "vehicle": {"val": 1, "test": 1},
}

CATEGORY_ORDER = ["animals", "obstacles", "cones", "lost_tires", "vehicle"]
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLORS = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
CATEGORY_COLORS = {
    "animals": "#4C78A8",
    "obstacles": "#E45756",
    "cones": "#F58518",
    "lost_tires": "#72B7B2",
    "vehicle": "#B279A2",
}


@dataclass
class FrameRecord:
    frame_id: str
    category: str
    scene_group: str
    image_path: Path
    semantic_path: Path
    instance_path: Path | None
    index_path: Path | None
    width: int
    height: int
    anomaly_pixels: int
    total_pixels: int
    label_values: list[int]
    n_instances: int | None
    instance_agrees: bool | None


def category_of(frame_id: str) -> str:
    """Map a file stem such as animals01_... or lost_tires03 to a category."""
    if frame_id.startswith("lost_tires"):
        return "lost_tires"
    prefix = frame_id.split("_", 1)[0]
    return re.sub(r"\d+$", "", prefix)


def scene_group_of(frame_id: str) -> str:
    for group_name, members in DUPLICATE_SCENES.items():
        if frame_id in members:
            return group_name
    return frame_id


def load_semantic_mask(path: Path) -> np.ndarray:
    """Return a 2-D array of class ids."""
    array = np.array(Image.open(path))
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        rgb = array[..., :3]
        if np.all(rgb[..., 0] == rgb[..., 1]) and np.all(rgb[..., 1] == rgb[..., 2]):
            return rgb[..., 0]
    raise ValueError(f"Unexpected semantic-mask layout in {path}: shape {array.shape}")


def discover_frames(dataset_root: Path) -> list[FrameRecord]:
    """Pair every listed image with its semantic mask and audit the pair."""
    frames_dir = dataset_root / "frames"
    listed = json.loads((dataset_root / "frame_list.json").read_text(encoding="utf-8"))
    records: list[FrameRecord] = []

    for file_name in listed:
        frame_id = file_name.removesuffix(".webp")
        image_path = frames_dir / file_name
        label_dir = frames_dir / f"{frame_id}.labels"
        semantic_path = label_dir / "labels_semantic.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Listed image is missing: {image_path}")
        if not semantic_path.exists():
            raise FileNotFoundError(f"Semantic mask is missing for {frame_id}")

        with Image.open(image_path) as image:
            width, height = image.size
        mask = load_semantic_mask(semantic_path)
        if mask.shape != (height, width):
            raise ValueError(
                f"Size mismatch for {frame_id}: image {width}x{height}, "
                f"mask {mask.shape[1]}x{mask.shape[0]}"
            )

        values, counts = np.unique(mask, return_counts=True)
        count_of = {int(v): int(c) for v, c in zip(values, counts)}
        unexpected = sorted(set(count_of) - {BACKGROUND_LABEL, ANOMALY_LABEL})
        if unexpected:
            raise ValueError(f"{frame_id} has unexpected label values {unexpected}")

        instance_path = label_dir / "labels_instance.png"
        index_path = label_dir / "index.json"
        n_instances = None
        if index_path.exists():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            n_instances = len(payload.get("instances", []))

        instance_agrees = None
        if instance_path.exists():
            instance = np.array(Image.open(instance_path))
            if instance.shape[:2] != mask.shape:
                instance_agrees = False
            else:
                anomaly = mask == ANOMALY_LABEL
                background = mask == BACKGROUND_LABEL
                leaked = ((anomaly & (instance == 0)) | (background & (instance != 0))).any()
                instance_agrees = not bool(leaked)

        records.append(
            FrameRecord(
                frame_id=frame_id,
                category=category_of(frame_id),
                scene_group=scene_group_of(frame_id),
                image_path=image_path,
                semantic_path=semantic_path,
                instance_path=instance_path if instance_path.exists() else None,
                index_path=index_path if index_path.exists() else None,
                width=width,
                height=height,
                anomaly_pixels=count_of.get(ANOMALY_LABEL, 0),
                total_pixels=int(mask.size),
                label_values=sorted(count_of),
                n_instances=n_instances,
                instance_agrees=instance_agrees,
            )
        )
    return records


def records_to_frame(records: list[FrameRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "frame_id": record.frame_id,
                "category": record.category,
                "scene_group": record.scene_group,
                "width": record.width,
                "height": record.height,
                "megapixels": record.total_pixels / 1_000_000,
                "anomaly_pixels": record.anomaly_pixels,
                "total_pixels": record.total_pixels,
                "anomaly_ratio": record.anomaly_pixels / record.total_pixels,
                "n_instances": record.n_instances,
                "has_instance_mask": record.instance_path is not None,
                "has_instance_index": record.index_path is not None,
                "instance_agrees_with_semantic": record.instance_agrees,
                "image_path": str(record.image_path),
                "semantic_path": str(record.semantic_path),
            }
        )
    return pd.DataFrame(rows)


def _evenly_spaced(n: int, k: int) -> list[int]:
    """Return k distinct indexes spread from the first item to the last."""
    if k == 0:
        return []
    if k > n:
        raise ValueError(f"Cannot pick {k} items from {n}")
    if k == 1:
        return [n // 2]
    raw = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
    chosen: list[int] = []
    used: set[int] = set()
    for index in raw:
        if index in used:
            for distance in range(1, n):
                for candidate in (index - distance, index + distance):
                    if 0 <= candidate < n and candidate not in used:
                        index = candidate
                        break
                else:
                    continue
                break
        used.add(index)
        chosen.append(index)
    return sorted(chosen)


def _assign_category(frame_ids: list[str], ratios: dict[str, float], n_val: int, n_test: int) -> dict[str, str]:
    """Spread hold-out images across small, medium, and large obstacles.

    Images are ranked by anomaly area. Hold-out ranks are evenly spaced
    along that ranking, then alternate between test and validation.
    """
    ordered = sorted(frame_ids, key=lambda frame_id: (ratios[frame_id], frame_id))
    n = len(ordered)
    if n_val + n_test > n:
        raise ValueError(f"Hold-out quota {n_val}+{n_test} exceeds {n} images")

    positions = _evenly_spaced(n, n_val + n_test)
    test_positions: list[int] = []
    val_positions: list[int] = []
    for turn, position in enumerate(positions):
        give_to_test = turn % 2 == 0
        if give_to_test and len(test_positions) < n_test:
            test_positions.append(position)
        elif len(val_positions) < n_val:
            val_positions.append(position)
        else:
            test_positions.append(position)

    held_out = set(test_positions) | set(val_positions)
    assignment = {ordered[i]: "train" for i in range(n) if i not in held_out}
    assignment.update({ordered[i]: "val" for i in val_positions})
    assignment.update({ordered[i]: "test" for i in test_positions})
    return assignment


def assign_splits(inventory: pd.DataFrame) -> pd.DataFrame:
    """Assign each frame to train, val, or test without splitting a scene."""
    ratios = dict(zip(inventory["frame_id"], inventory["anomaly_ratio"]))
    duplicate_ids = {
        frame_id
        for members in DUPLICATE_SCENES.values()
        for frame_id in members
    }
    missing = duplicate_ids - set(inventory["frame_id"])
    if missing:
        raise ValueError(f"Duplicate-scene ids not in the dataset: {sorted(missing)}")

    assignment: dict[str, str] = {frame_id: "train" for frame_id in duplicate_ids}
    independent = inventory[~inventory["frame_id"].isin(duplicate_ids)]

    for category, quota in HOLDOUT_QUOTAS.items():
        members = independent.loc[independent["category"] == category, "frame_id"].tolist()
        assignment.update(_assign_category(members, ratios, quota["val"], quota["test"]))

    leftover = set(inventory["frame_id"]) - set(assignment)
    if leftover:
        raise ValueError(f"Frames were not assigned a split: {sorted(leftover)}")

    assigned = inventory.copy()
    assigned["split"] = assigned["frame_id"].map(assignment)
    return assigned.sort_values(["split", "category", "anomaly_ratio", "frame_id"]).reset_index(drop=True)


def binary_mask(semantic: np.ndarray) -> Image.Image:
    mask = np.zeros(semantic.shape, dtype=np.uint8)
    mask[semantic == ANOMALY_LABEL] = ANOMALY_PIXEL
    return Image.fromarray(mask, mode="L")


def letterbox(image: Image.Image, mask: Image.Image, size: int = CANVAS_SIZE) -> dict:
    """Scale uniformly so the long side fits, then pad to a square canvas.

    Images use bilinear resampling. Masks use nearest-neighbour resampling so
    class ids stay exact.
    """
    width, height = image.size
    scale = size / max(width, height)
    resized_w = min(size, max(1, int(round(width * scale))))
    resized_h = min(size, max(1, int(round(height * scale))))
    image_r = image.convert("RGB").resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    mask_r = mask.resize((resized_w, resized_h), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    mask_canvas = Image.new("L", (size, size), 0)
    pad_left = (size - resized_w) // 2
    pad_top = (size - resized_h) // 2
    canvas.paste(image_r, (pad_left, pad_top))
    mask_canvas.paste(mask_r, (pad_left, pad_top))
    anomaly_pixels = int(np.array(mask_canvas).sum() // ANOMALY_PIXEL)
    return {
        "image": canvas,
        "mask": mask_canvas,
        "scale": scale,
        "resized_width": resized_w,
        "resized_height": resized_h,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "anomaly_pixels": anomaly_pixels,
    }


def prepare_splits(assigned: pd.DataFrame, output_dir: Path, size: int = CANVAS_SIZE) -> pd.DataFrame:
    """Write letterboxed images and binary masks, and return the manifest."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in SPLIT_ORDER:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    rows = []
    for record in assigned.itertuples(index=False):
        semantic = load_semantic_mask(Path(record.semantic_path))
        source = binary_mask(semantic)
        with Image.open(record.image_path) as image:
            packed = letterbox(image, source, size=size)
        if packed["anomaly_pixels"] == 0:
            raise RuntimeError(
                f"Resizing erased the anomaly in {record.frame_id}. "
                "Use a larger canvas before training."
            )
        image_name = f"{record.frame_id}.png"
        image_out = output_dir / record.split / "images" / image_name
        mask_out = output_dir / record.split / "masks" / image_name
        packed["image"].save(image_out)
        packed["mask"].save(mask_out)
        rows.append(
            {
                "frame_id": record.frame_id,
                "category": record.category,
                "scene_group": record.scene_group,
                "split": record.split,
                "original_width": record.width,
                "original_height": record.height,
                "anomaly_pixels_original": record.anomaly_pixels,
                "anomaly_ratio_original": record.anomaly_ratio,
                "n_instances": record.n_instances,
                "canvas_size": size,
                "scale": packed["scale"],
                "resized_width": packed["resized_width"],
                "resized_height": packed["resized_height"],
                "pad_left": packed["pad_left"],
                "pad_top": packed["pad_top"],
                "anomaly_pixels_processed": packed["anomaly_pixels"],
                "anomaly_ratio_processed": packed["anomaly_pixels"] / (size * size),
                "image_path": str(image_out),
                "mask_path": str(mask_out),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    return manifest


def overlay_anomaly(image: np.ndarray, mask: np.ndarray, color=(220, 30, 45), alpha=0.45) -> np.ndarray:
    """Tint anomalous pixels so the ground-truth region is easy to see."""
    painted = image.copy()
    region = mask > 0
    if not region.any():
        return painted
    tint = np.array(color, dtype=np.float32)
    pixels = painted[region].astype(np.float32)
    painted[region] = np.clip(pixels * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
    return painted


def _style_axes(ax, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=13, pad=10, fontweight="medium")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", zorder=0)
    ax.set_axisbelow(True)


def save_figures(inventory: pd.DataFrame, manifest: pd.DataFrame, figure_dir: Path) -> dict[str, Path]:
    """Write the figures used in the report. Returns their file paths."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    paths: dict[str, Path] = {}

    counts = inventory["category"].value_counts().reindex(CATEGORY_ORDER)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[CATEGORY_COLORS[c] for c in counts.index],
        zorder=2,
    )
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, str(int(value)), ha="center", va="bottom")
    ax.set_ylim(0, counts.max() + 6)
    _style_axes(ax, "Most scenes show animals; cones, rocks, and vehicles are rarer", "Obstacle category", "Number of images")
    paths["categories"] = figure_dir / "01_category_counts.png"
    fig.savefig(paths["categories"], dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), gridspec_kw={"width_ratios": [1.0, 1.7]})
    axes[0].axis("off")
    axes[0].text(0.0, 0.72, "1280 × 720", fontsize=26, fontweight="bold", color="#222222")
    axes[0].text(0.0, 0.52, "Every one of the 60 images", fontsize=13, color="#333333")
    axes[0].text(
        0.0,
        0.28,
        "16:9 road scenes\n921,600 pixels each\nLabels use the same grid",
        fontsize=12,
        color="#555555",
        linespacing=1.5,
    )
    ratios = inventory["anomaly_ratio"] * 100
    axes[1].hist(ratios, bins=12, color="#4C78A8", edgecolor="white", zorder=2)
    median = float(ratios.median())
    axes[1].axvline(median, color="#E45756", linestyle="--", linewidth=1.5, label=f"Median {median:.1f}%", zorder=3)
    axes[1].legend(frameon=False)
    _style_axes(
        axes[1],
        "Obstacle size is what varies",
        "Anomalous pixels in the image (%)",
        "Number of images",
    )
    fig.suptitle("Resolution is fixed. The hard part is how small the obstacle can be.", fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    paths["resolutions"] = figure_dir / "02_resolution_and_obstacle_size.png"
    fig.savefig(paths["resolutions"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    data = [inventory.loc[inventory["category"] == c, "anomaly_ratio"] * 100 for c in CATEGORY_ORDER]
    box = ax.boxplot(data, tick_labels=CATEGORY_ORDER, patch_artist=True, medianprops={"color": "black"})
    for patch, category in zip(box["boxes"], CATEGORY_ORDER):
        patch.set_facecolor(CATEGORY_COLORS[category])
        patch.set_alpha(0.85)
    _style_axes(
        ax,
        "The obstacle usually covers a small fraction of the image",
        "Obstacle category",
        "Anomalous pixels (%)",
    )
    paths["anomaly_ratio"] = figure_dir / "03_anomaly_area_by_category.png"
    fig.savefig(paths["anomaly_ratio"], dpi=160)
    plt.close(fig)

    anomaly_share = inventory["anomaly_pixels"].sum() / inventory["total_pixels"].sum()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    shares = [1 - anomaly_share, anomaly_share]
    bars = ax.bar(["Background", "Anomaly"], [s * 100 for s in shares], color=["#B8B8B8", "#E45756"], zorder=2)
    for bar, share in zip(bars, shares):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{share * 100:.1f}%",
            ha="center",
        )
    ax.set_ylim(0, 118)
    _style_axes(ax, "A pixel chosen at random is almost always background", None, "Share of all pixels (%)")
    fig.text(
        0.5,
        -0.04,
        "Labelling every pixel as background is about 90% accurate and still misses every obstacle (IoU = 0).",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    paths["class_balance"] = figure_dir / "04_pixel_class_balance.png"
    fig.savefig(paths["class_balance"], dpi=160)
    plt.close(fig)

    paths["examples"] = figure_dir / "05_image_mask_overlay_examples.png"
    _save_example_grid(inventory, paths["examples"])

    paths["size_extremes"] = figure_dir / "06_small_and_large_anomalies.png"
    _save_extremes(inventory, paths["size_extremes"])

    counts = (
        manifest.groupby(["split", "category"]).size().unstack(fill_value=0).reindex(index=SPLIT_ORDER, columns=CATEGORY_ORDER)
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    left = np.zeros(len(SPLIT_ORDER))
    label_color = {"animals": "white", "obstacles": "white", "cones": "#3a2a00", "lost_tires": "#143330", "vehicle": "white"}
    for category in CATEGORY_ORDER:
        values = counts[category].to_numpy()
        ax.barh(SPLIT_ORDER, values, left=left, color=CATEGORY_COLORS[category], label=category, zorder=2)
        for row, (value, start) in enumerate(zip(values, left)):
            if value >= 1:
                ax.text(
                    start + value / 2,
                    row,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color=label_color[category],
                    fontsize=9,
                    fontweight="medium",
                    zorder=3,
                )
        left = left + values
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Category")
    ax.set_xlabel("Number of images")
    ax.set_title("42 images for training, 9 for validation, 9 for testing", loc="left", fontsize=13, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#E6E6E6")
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    fig.tight_layout()
    paths["splits"] = figure_dir / "07_split_composition.png"
    fig.savefig(paths["splits"], dpi=160)
    plt.close(fig)

    paths["preprocess"] = figure_dir / "08_preprocessing_letterbox.png"
    save_preprocess_figure(inventory, manifest, paths["preprocess"])

    paths["contact_sheet"] = figure_dir / "09_all_scenes_by_split.png"
    _save_contact_sheet(manifest, paths["contact_sheet"])
    paths["repeated_scenes"] = figure_dir / "10_repeated_scenes.png"
    _save_repeated_scenes(inventory, paths["repeated_scenes"])
    paths["correspondence"] = figure_dir / "11_mask_correspondence_zoom.png"
    _save_correspondence_zoom(inventory, paths["correspondence"])
    return paths


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def _panel(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


# Clear teaching examples. The lost-tire median frame marks a person in the lane,
# which is a confusing first picture of that category. The tire itself is shown instead.
SHOWCASE_FRAMES = {
    "lost_tires": "lost_tires08",
}


def _save_example_grid(inventory: pd.DataFrame, path: Path) -> None:
    chosen = []
    for category in CATEGORY_ORDER:
        if category in SHOWCASE_FRAMES:
            chosen.append(inventory.loc[inventory["frame_id"] == SHOWCASE_FRAMES[category]].iloc[0])
            continue
        part = inventory[inventory["category"] == category].sort_values("anomaly_ratio")
        chosen.append(part.iloc[len(part) // 2])

    fig, axes = plt.subplots(len(chosen), 3, figsize=(9.6, 13.5))
    for row, record in enumerate(chosen):
        image = _read_rgb(Path(record.image_path))
        mask = np.array(binary_mask(load_semantic_mask(Path(record.semantic_path))))
        display_mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
        _panel(axes[row, 0], image, record.category if row == 0 else "")
        _panel(axes[row, 1], display_mask, "Ground-truth mask" if row == 0 else "")
        _panel(axes[row, 2], overlay_anomaly(image, mask), "Mask placed on the image" if row == 0 else "")
        axes[row, 0].set_ylabel(record.category, fontsize=11)
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(record.category, fontsize=11)
        percent = record.anomaly_ratio * 100
        axes[row, 2].text(
            0.02,
            0.04,
            f"{percent:.1f}% of pixels",
            transform=axes[row, 2].transAxes,
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
        )
    fig.suptitle("Each image has a mask that marks the unexpected obstacle", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_extremes(inventory: pd.DataFrame, path: Path) -> None:
    ordered = inventory.sort_values("anomaly_ratio")
    picks = [ordered.iloc[0], ordered.iloc[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for row, record in enumerate(picks):
        image = _read_rgb(Path(record.image_path))
        mask = np.array(binary_mask(load_semantic_mask(Path(record.semantic_path))))
        label = "Smallest anomaly" if row == 0 else "Largest anomaly"
        _panel(axes[row, 0], image, f"{label}: {record.frame_id}")
        _panel(axes[row, 1], overlay_anomaly(image, mask), f"{record.anomaly_ratio * 100:.2f}% of the image")
    fig.suptitle("Difficulty varies sharply: some obstacles are tiny, others fill the view", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_preprocess_figure(inventory: pd.DataFrame, manifest: pd.DataFrame, path: Path) -> None:
    """Show one image before and after the square-canvas preparation."""
    record = manifest.sort_values("anomaly_ratio_original").iloc[len(manifest) // 2]
    source = inventory.loc[inventory["frame_id"] == record.frame_id].iloc[0]
    original_image = _read_rgb(Path(source.image_path))
    original_mask = np.array(binary_mask(load_semantic_mask(Path(source.semantic_path))))
    prepared_image = _read_rgb(Path(record.image_path))
    prepared_mask = np.array(Image.open(record.mask_path))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    _panel(axes[0, 0], original_image, f"Original {source.width}×{source.height}")
    _panel(axes[0, 1], overlay_anomaly(original_image, original_mask), "Original mask on the image")
    _panel(axes[1, 0], prepared_image, f"Letterboxed to {CANVAS_SIZE}×{CANVAS_SIZE}")
    _panel(axes[1, 1], overlay_anomaly(prepared_image, prepared_mask), "Mask transformed with the image")
    fig.suptitle(
        "The image keeps its shape; empty borders make every sample the same size",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_contact_sheet(manifest: pd.DataFrame, path: Path) -> None:
    ordered = manifest.sort_values(["split", "category", "frame_id"])
    columns = 6
    rows = int(np.ceil(len(ordered) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, rows * 2.15))
    for ax, record in zip(axes.ravel(), ordered.itertuples(index=False)):
        image = _read_rgb(Path(record.image_path))
        mask = np.array(Image.open(record.mask_path))
        ax.imshow(overlay_anomaly(image, mask, alpha=0.5))
        ax.set_title(f"{record.split}  ·  {record.category}", color=SPLIT_COLORS[record.split], fontsize=8, loc="left")
        for spine in ax.spines.values():
            spine.set_color(SPLIT_COLORS[record.split])
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(ordered) :]:
        ax.axis("off")
    fig.suptitle(
        "All 60 prepared scenes. Green border = test, orange = validation, blue = training.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of the biggest connected obstacle. Falls back to the whole mask."""
    region = mask > 0
    if not region.any():
        return region
    try:
        from scipy import ndimage
    except ImportError:
        return region
    labels, count = ndimage.label(region)
    if count <= 1:
        return region
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def _crop_to_region(image: np.ndarray, mask: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(region)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad = max(16, int(round(0.45 * max(y1 - y0 + 1, x1 - x0 + 1))))
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(image.shape[0] - 1, y1 + pad)
    x1 = min(image.shape[1] - 1, x1 + pad)
    return image[y0 : y1 + 1, x0 : x1 + 1], mask[y0 : y1 + 1, x0 : x1 + 1]


def _outline(mask: np.ndarray) -> np.ndarray:
    """Pixels on the boundary of the obstacle, thickened so the line is visible."""
    region = mask > 0
    interior = region.copy()
    interior[1:] &= region[:-1]
    interior[:-1] &= region[1:]
    interior[:, 1:] &= region[:, :-1]
    interior[:, :-1] &= region[:, 1:]
    edge = region & ~interior
    thick = edge.copy()
    thick[1:] |= edge[:-1]
    thick[:-1] |= edge[1:]
    thick[:, 1:] |= edge[:, :-1]
    thick[:, :-1] |= edge[:, 1:]
    return thick


def _save_correspondence_zoom(inventory: pd.DataFrame, path: Path) -> None:
    """Zoom on four objects so the mask boundary can be compared with the object edge."""
    cases = [
        ("lost_tires08", "Loose tire"),
        ("cones03_Bang_Na_Trat_Highway", "Traffic cone"),
        ("animals17_E6_near_Kvaenangen", "Reindeer"),
        ("obstacles03_cardboard_czech_hedgehog", "Barrier"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.6))
    for column, (frame_id, caption) in enumerate(cases):
        record = inventory.loc[inventory["frame_id"] == frame_id].iloc[0]
        image = _read_rgb(Path(record.image_path))
        mask = np.array(binary_mask(load_semantic_mask(Path(record.semantic_path))))
        crop_image, crop_mask = _crop_to_region(image, mask, _largest_component(mask))
        painted = overlay_anomaly(crop_image, crop_mask, alpha=0.35)
        painted[_outline(crop_mask)] = (255, 220, 0)
        percent = record.anomaly_ratio * 100
        _panel(axes[0, column], crop_image, caption)
        _panel(axes[1, column], painted, f"whole image: {percent:.1f}% obstacle")
    fig.suptitle(
        "The yellow line is the edge of the ground-truth mask. It follows the object.",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.01,
        "Top: the photograph, cropped around the obstacle.  Bottom: the same crop with the mask tinted and outlined.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_repeated_scenes(inventory: pd.DataFrame, path: Path) -> None:
    """Show the three source sequences that must stay inside one split."""
    groups = [
        ("Same dashcam video: a loose tire", DUPLICATE_SCENES["lost_tires_sequence"]),
        ("Same sculpture, several photographs", DUPLICATE_SCENES["cardboard_czech_hedgehog"]),
        ("Same rockfall sequence", DUPLICATE_SCENES["rocks_sequence"]),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 8.2))
    for row, (title, members) in enumerate(groups):
        for col in range(4):
            ax = axes[row, col]
            if col >= len(members):
                ax.axis("off")
                continue
            record = inventory.loc[inventory["frame_id"] == members[col]].iloc[0]
            image = _read_rgb(Path(record.image_path))
            mask = np.array(binary_mask(load_semantic_mask(Path(record.semantic_path))))
            ax.imshow(overlay_anomaly(image, mask, alpha=0.4))
            if members[col].startswith("lost_tires"):
                short = members[col]
            elif "rocks" in members[col]:
                short = "rocks" + members[col].rsplit("rocks", 1)[-1]
            else:
                short = members[col].split("_")[0]
            ax.set_title(short, fontsize=9)
            ax.axis("off")
        axes[row, 0].text(
            -0.08,
            0.5,
            title,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
            color="#333333",
        )
    fig.suptitle("These frames show one event more than once, so they all stay in training", fontsize=13)
    fig.tight_layout(rect=(0.06, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summarize(inventory: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Numbers quoted by the written report."""
    by_category = (
        inventory.groupby("category")
        .agg(
            images=("frame_id", "size"),
            anomaly_ratio_median=("anomaly_ratio", "median"),
            anomaly_ratio_mean=("anomaly_ratio", "mean"),
            width_min=("width", "min"),
            width_max=("width", "max"),
            height_min=("height", "min"),
            height_max=("height", "max"),
        )
        .reindex(CATEGORY_ORDER)
    )
    split_counts = manifest.groupby(["split", "category"]).size().unstack(fill_value=0)
    return {
        "n_images": int(len(inventory)),
        "n_sizes": int(inventory[["width", "height"]].drop_duplicates().shape[0]),
        "width_min": int(inventory["width"].min()),
        "width_max": int(inventory["width"].max()),
        "height_min": int(inventory["height"].min()),
        "height_max": int(inventory["height"].max()),
        "anomaly_ratio_min": float(inventory["anomaly_ratio"].min()),
        "anomaly_ratio_median": float(inventory["anomaly_ratio"].median()),
        "anomaly_ratio_mean": float(inventory["anomaly_ratio"].mean()),
        "anomaly_ratio_max": float(inventory["anomaly_ratio"].max()),
        "pixel_anomaly_share": float(inventory["anomaly_pixels"].sum() / inventory["total_pixels"].sum()),
        "smallest_frame": inventory.loc[inventory["anomaly_ratio"].idxmin(), "frame_id"],
        "largest_frame": inventory.loc[inventory["anomaly_ratio"].idxmax(), "frame_id"],
        "instance_masks": int(inventory["has_instance_mask"].sum()),
        "instance_indexes": int(inventory["has_instance_index"].sum()),
        "instance_disagreements": int((inventory["instance_agrees_with_semantic"] == False).sum()),  # noqa: E712
        "categories": {k: int(v) for k, v in inventory["category"].value_counts().to_dict().items()},
        "by_category": by_category.reset_index().to_dict(orient="records"),
        "split_counts": {split: int((manifest["split"] == split).sum()) for split in SPLIT_ORDER},
        "split_by_category": split_counts.reindex(index=SPLIT_ORDER, columns=CATEGORY_ORDER).fillna(0).astype(int).to_dict(),
        "processed_anomaly_ratio_min": float(manifest["anomaly_ratio_processed"].min()),
        "processed_anomaly_pixels_min": int(manifest["anomaly_pixels_processed"].min()),
        "canvas_size": CANVAS_SIZE,
        "letterbox_scale": 512 / 1280,
        "letterbox_content": [512, 288],
        "letterbox_pad_top": 112,
        "scene_groups_crossing_splits": int(
            manifest.groupby("scene_group")["split"].nunique().gt(1).sum()
        ),
        "frames_without_instance_mask": inventory.loc[
            ~inventory["has_instance_mask"], "frame_id"
        ].tolist(),
        "split_anomaly_ratio": {
            split: {
                "min": float(part["anomaly_ratio_original"].min()),
                "median": float(part["anomaly_ratio_original"].median()),
                "max": float(part["anomaly_ratio_original"].max()),
            }
            for split, part in manifest.groupby("split")
        },
    }


def run(dataset_root: Path, output_dir: Path, figure_dir: Path, table_dir: Path) -> dict:
    records = discover_frames(dataset_root)
    inventory = records_to_frame(records)
    assigned = assign_splits(inventory)
    manifest = prepare_splits(assigned, output_dir, size=CANVAS_SIZE)
    table_dir.mkdir(parents=True, exist_ok=True)
    inventory.sort_values(["category", "frame_id"]).to_csv(table_dir / "frame_inventory.csv", index=False)
    assigned.sort_values(["split", "category", "frame_id"]).to_csv(table_dir / "split_assignment.csv", index=False)
    paths = save_figures(inventory, manifest, figure_dir)
    summary = summarize(inventory, manifest)
    summary["figures"] = {name: str(path) for name, path in paths.items()}
    (output_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (table_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
