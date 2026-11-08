"""Challenge pool — training (train/val GT only) and discovery (no GT)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

# Absolute project root (RoadAnomaly/)
ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_SPLITS = frozenset({"test"})
ALLOWED_TRAINING_SPLITS = frozenset({"train", "val"})


class ChallengeType(str, Enum):
    TRAINING = "training"
    DISCOVERY = "discovery"


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    challenge_type: ChallengeType
    frame_id: str | None
    image_path: Path | None
    mask_path: Path | None
    split: str | None
    category: str | None
    license_note: str
    source_note: str

    def load_rgb(self) -> np.ndarray:
        if self.image_path is None or not self.image_path.is_file():
            raise FileNotFoundError("Image du défi introuvable.")
        return np.asarray(Image.open(self.image_path).convert("RGB"), dtype=np.uint8)


def _credits_blurb(frame_id: str) -> tuple[str, str]:
    credits = ROOT / "credits.txt"
    note = "Images RoadAnomaly — usage recherche ; annotations CC BY-SA 4.0 (voir credits.txt)."
    source = f"RoadAnomaly frame `{frame_id}`"
    if credits.is_file():
        text = credits.read_text(encoding="utf-8", errors="replace")
        # Best-effort: include filename mention if present nearby
        if frame_id in text:
            source = f"RoadAnomaly / credits.txt mention for {frame_id}"
    return note, source


def _resolve_frame_paths(frame_id: str) -> tuple[Path | None, Path | None]:
    """Prefer original frames/; fall back to processed PNGs."""
    webp = ROOT / "frames" / f"{frame_id}.webp"
    label = ROOT / "frames" / f"{frame_id}.labels" / "labels_semantic.png"
    if webp.is_file() and label.is_file():
        return webp, label
    # processed fallbacks — search train then val only
    for split in ("train", "val"):
        img = ROOT / "data" / "processed" / split / "images" / f"{frame_id}.png"
        msk = ROOT / "data" / "processed" / split / "masks" / f"{frame_id}.png"
        if img.is_file() and msk.is_file():
            return img, msk
    return None, None


def load_training_challenges(
    manifest_path: Path | None = None,
    *,
    root: Path | None = None,
) -> list[Challenge]:
    """Load challenges with verified GT from train/val only. Never includes test."""
    root = root or ROOT
    manifest_path = manifest_path or (root / "data" / "processed" / "split_manifest.csv")
    if not manifest_path.is_file():
        return []

    df = pd.read_csv(manifest_path)
    if "split" not in df.columns or "frame_id" not in df.columns:
        raise ValueError("split_manifest.csv invalide (colonnes split/frame_id manquantes).")

    # Hard block test rows
    blocked = df.loc[df["split"].astype(str).str.lower().isin(FORBIDDEN_SPLITS)]
    if len(blocked) and set(df["split"].unique()) <= FORBIDDEN_SPLITS:
        raise RuntimeError("Manifest ne contient que le split test — pool jeu interdit.")

    allowed = df.loc[df["split"].astype(str).str.lower().isin(ALLOWED_TRAINING_SPLITS)].copy()
    challenges: list[Challenge] = []
    for row in allowed.itertuples(index=False):
        frame_id = str(row.frame_id)
        split = str(row.split).lower()
        if split in FORBIDDEN_SPLITS:
            continue
        image_path, mask_path = _resolve_frame_paths(frame_id)
        if image_path is None or mask_path is None:
            continue
        license_note, source_note = _credits_blurb(frame_id)
        category = getattr(row, "category", None)
        challenges.append(
            Challenge(
                challenge_id=f"train::{frame_id}",
                challenge_type=ChallengeType.TRAINING,
                frame_id=frame_id,
                image_path=image_path,
                mask_path=mask_path,
                split=split,
                category=str(category) if category is not None else None,
                license_note=license_note,
                source_note=source_note,
            )
        )
    return challenges


def pick_challenge(
    challenges: list[Challenge],
    *,
    exclude_ids: set[str] | None = None,
    seed: int | None = None,
) -> Challenge | None:
    """Pick a random unused training challenge."""
    pool = [c for c in challenges if not exclude_ids or c.challenge_id not in exclude_ids]
    if not pool:
        pool = list(challenges)
    if not pool:
        return None
    rng = np.random.default_rng(seed)
    return pool[int(rng.integers(0, len(pool)))]


def content_fingerprint(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def discovery_meta_from_upload(
    *,
    filename: str,
    image_bytes: bytes,
    license_claim: str,
    source_claim: str,
) -> dict[str, Any]:
    return {
        "challenge_type": ChallengeType.DISCOVERY.value,
        "filename": filename,
        "content_sha256": content_fingerprint(image_bytes),
        "license_claim": license_claim.strip(),
        "source_claim": source_claim.strip(),
        "byte_size": len(image_bytes),
    }


def assert_not_test_frame(frame_id: str, manifest_path: Path | None = None) -> None:
    """Raise if frame_id belongs to the locked test split."""
    manifest_path = manifest_path or (ROOT / "data" / "processed" / "split_manifest.csv")
    df = pd.read_csv(manifest_path)
    hit = df.loc[df["frame_id"].astype(str) == frame_id]
    if hit.empty:
        return
    split = str(hit.iloc[0]["split"]).lower()
    if split == "test":
        raise PermissionError(
            f"Frame `{frame_id}` appartient au split test verrouillé — usage jeu interdit."
        )


def pool_summary_json(challenges: list[Challenge]) -> str:
    payload = {
        "n_training": len(challenges),
        "splits": sorted({c.split for c in challenges if c.split}),
        "forbidden_splits": sorted(FORBIDDEN_SPLITS),
    }
    return json.dumps(payload, indent=2)
