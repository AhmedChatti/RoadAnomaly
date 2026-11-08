"""Mask helpers for Anomaly Hunter (alignment, validation, canvas conversion)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


MAX_IMAGE_PIXELS = 12_000_000  # ~4k×3k safety bound
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
DEFAULT_TILE_GRID = 3


def tile_bounds(image_hw: tuple[int, int], grid: int, row: int, col: int) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) for one grid cell."""
    h, w = image_hw
    if grid < 2:
        raise ValueError("grid must be >= 2")
    y0 = row * h // grid
    y1 = (row + 1) * h // grid
    x0 = col * w // grid
    x1 = (col + 1) * w // grid
    return y0, y1, x0, x1


def mask_from_selected_tiles(
    image_hw: tuple[int, int],
    selected: set[tuple[int, int]] | list[tuple[int, int]],
    *,
    grid: int = DEFAULT_TILE_GRID,
) -> np.ndarray:
    """Build a binary mask by filling selected grid rectangles (reCAPTCHA-style)."""
    h, w = image_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for row, col in selected:
        if not (0 <= row < grid and 0 <= col < grid):
            continue
        y0, y1, x0, x1 = tile_bounds((h, w), grid, int(row), int(col))
        mask[y0:y1, x0:x1] = 1
    return mask


def overlay_tile_grid_rgb(
    image_rgb: np.ndarray,
    selected: set[tuple[int, int]] | list[tuple[int, int]],
    *,
    grid: int = DEFAULT_TILE_GRID,
) -> np.ndarray:
    """Preview image with grid lines and highlighted selected tiles."""
    img = np.asarray(image_rgb, dtype=np.uint8).copy()
    h, w = img.shape[:2]
    selected_set = {(int(r), int(c)) for r, c in selected}
    # Highlight selected cells
    for row, col in selected_set:
        y0, y1, x0, x1 = tile_bounds((h, w), grid, row, col)
        tint = img[y0:y1, x0:x1].astype(np.float32)
        color = np.array([232, 93, 42], dtype=np.float32)
        tint = 0.55 * tint + 0.45 * color
        img[y0:y1, x0:x1] = np.clip(tint, 0, 255).astype(np.uint8)
    # Grid lines
    for i in range(1, grid):
        y = i * h // grid
        x = i * w // grid
        img[max(0, y - 1) : y + 2, :, :] = (255, 255, 255)
        img[:, max(0, x - 1) : x + 2, :] = (255, 255, 255)
    return img


def remap_semantic_to_binary(mask: np.ndarray) -> np.ndarray:
    """Map RoadAnomaly semantic labels {0,2} or prepared {0,1,255} to {0,1}."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    flat = arr.astype(np.int32)
    values = set(int(v) for v in np.unique(flat))
    if values.issubset({0, 2}):
        return (flat == 2).astype(np.uint8)
    if values.issubset({0, 1}):
        return (flat > 0).astype(np.uint8)
    if values.issubset({0, 255}):
        return (flat > 0).astype(np.uint8)
    if values.issubset({0, 1, 255}):
        return (flat > 0).astype(np.uint8)
    raise ValueError(f"Unexpected mask label values: {sorted(values)}")


def load_ground_truth_mask(mask_path: Path, *, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Load a verified GT mask and optionally resize with nearest-neighbor to (H, W)."""
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth mask introuvable: {path}")
    arr = np.asarray(Image.open(path))
    binary = remap_semantic_to_binary(arr)
    if target_hw is not None:
        h, w = target_hw
        if binary.shape != (h, w):
            binary = np.asarray(
                Image.fromarray(binary * 255, mode="L").resize((w, h), Image.Resampling.NEAREST)
            )
            binary = (binary > 127).astype(np.uint8)
    return binary.astype(np.uint8)


def validate_player_mask(
    mask: np.ndarray,
    *,
    expected_hw: tuple[int, int],
    require_any_positive: bool = False,
) -> np.ndarray:
    """Ensure mask is binary uint8 and matches expected (H, W)."""
    if mask is None:
        raise ValueError("Masque manquant.")
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != expected_hw:
        raise ValueError(
            f"Dimensions du masque incorrectes: {arr.shape} (attendu {expected_hw})."
        )
    binary = (arr > 0).astype(np.uint8)
    if require_any_positive and int(binary.sum()) == 0:
        raise ValueError("Le masque est vide. Dessinez autour de l'obstacle avant de valider.")
    return binary


def _strokes_to_mask(
    strokes: list[dict[str, Any]],
    *,
    canvas_w: int,
    canvas_h: int,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Rasterize drawable-canvas stroke objects onto a binary mask at target size."""
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError("Dimensions du canvas invalides.")
    img = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        if not isinstance(stroke, dict):
            continue
        path = stroke.get("path") or []
        points: list[tuple[float, float]] = []
        for cmd in path:
            if not cmd:
                continue
            op = cmd[0]
            if op in ("M", "L") and len(cmd) >= 3:
                points.append((float(cmd[1]), float(cmd[2])))
            elif op == "Q" and len(cmd) >= 5:
                points.append((float(cmd[3]), float(cmd[4])))
            elif op == "C" and len(cmd) >= 7:
                points.append((float(cmd[5]), float(cmd[6])))
        if len(points) < 2:
            continue
        stroke_type = str(stroke.get("type") or "freedraw").lower()
        color = stroke.get("stroke") or stroke.get("color") or "#ffffff"
        # Eraser strokes are typically white on black in canvas background workflows;
        # drawable-canvas uses stroke color; treat near-white / explicit eraser as erase.
        is_erase = (
            stroke_type in {"eraser", "erase"}
            or str(color).lower() in {"#ffffff", "#fff", "white", "rgba(255,255,255,1)"}
        )
        fill_val = 0 if is_erase else 255
        width = max(1, int(round(float(stroke.get("strokeWidth") or stroke.get("brushRadius") or 12))))
        if stroke_type in {"rect", "rectangle"} and len(points) >= 2:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            draw.rectangle([min(xs), min(ys), max(xs), max(ys)], outline=fill_val, width=width, fill=fill_val)
        elif stroke_type in {"polygon", "closed"} and len(points) >= 3:
            draw.polygon(points, outline=fill_val, fill=fill_val)
        else:
            draw.line(points, fill=fill_val, width=width, joint="curve")
            r = max(1, width // 2)
            for x, y in points:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=fill_val)

    canvas_mask = np.asarray(img, dtype=np.uint8)
    if (canvas_w, canvas_h) != (target_w, target_h):
        canvas_mask = np.asarray(
            Image.fromarray(canvas_mask, mode="L").resize(
                (target_w, target_h), Image.Resampling.NEAREST
            )
        )
    return (canvas_mask > 127).astype(np.uint8)


def binary_mask_from_canvas_json(
    canvas_result: dict[str, Any] | None,
    *,
    image_hw: tuple[int, int],
    canvas_width: int,
    canvas_height: int,
) -> np.ndarray:
    """Build a binary mask aligned to the original image from a drawable-canvas result."""
    h, w = image_hw
    if canvas_result is None:
        return np.zeros((h, w), dtype=np.uint8)

    # Prefer imageData if present (RGBA uint8), more faithful than stroke rebuild.
    image_data = canvas_result.get("imageData")
    if image_data is not None:
        try:
            arr = np.asarray(image_data)
            if arr.ndim == 3 and arr.shape[2] >= 4:
                # Alpha channel marks painted pixels on transparent background.
                painted = arr[..., 3] > 0
            elif arr.ndim == 3:
                painted = arr.max(axis=2) > 0
            else:
                painted = arr > 0
            painted_u8 = painted.astype(np.uint8) * 255
            if painted_u8.shape != (canvas_height, canvas_width):
                painted_u8 = np.asarray(
                    Image.fromarray(painted_u8, mode="L").resize(
                        (canvas_width, canvas_height), Image.Resampling.NEAREST
                    )
                )
            if (canvas_height, canvas_width) != (h, w):
                painted_u8 = np.asarray(
                    Image.fromarray(painted_u8, mode="L").resize((w, h), Image.Resampling.NEAREST)
                )
            return (painted_u8 > 127).astype(np.uint8)
        except Exception:
            pass

    objects = canvas_result.get("objects") or []
    if objects:
        return _strokes_to_mask(
            list(objects),
            canvas_w=canvas_width,
            canvas_h=canvas_height,
            target_w=w,
            target_h=h,
        )

    # json_data path from some canvas versions
    json_data = canvas_result.get("json_data") or {}
    objects = json_data.get("objects") or []
    if objects:
        return _strokes_to_mask(
            list(objects),
            canvas_w=canvas_width,
            canvas_h=canvas_height,
            target_w=w,
            target_h=h,
        )

    return np.zeros((h, w), dtype=np.uint8)


def overlay_mask_rgb(image_rgb: np.ndarray, mask: np.ndarray, color=(232, 93, 42), alpha: float = 0.45) -> np.ndarray:
    """Semi-transparent overlay for game comparison views."""
    img = np.asarray(image_rgb, dtype=np.float32).copy()
    m = (np.asarray(mask) > 0)
    if m.shape[:2] != img.shape[:2]:
        raise ValueError("Overlay shape mismatch")
    tint = np.zeros_like(img)
    tint[m] = np.array(color, dtype=np.float32)
    out = img.copy()
    out[m] = (1.0 - alpha) * img[m] + alpha * tint[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def difference_view(player: np.ndarray, other: np.ndarray) -> np.ndarray:
    """RGB diff: green=agreement, red=player-only, blue=other-only."""
    p = (np.asarray(player) > 0)
    o = (np.asarray(other) > 0)
    h, w = p.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    both = p & o
    only_p = p & ~o
    only_o = ~p & o
    vis[both] = (40, 180, 100)
    vis[only_p] = (220, 70, 60)
    vis[only_o] = (60, 110, 220)
    return vis
