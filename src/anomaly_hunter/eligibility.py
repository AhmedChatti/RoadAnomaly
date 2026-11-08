"""E3 training eligibility — approval alone is never enough.

Permanently excludes locked Stage-5 test split and Phase-1 validation images
from future E3 training exports. Does not train or modify E2b.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

# Permanent exclusions for any E3 training corpus derived from this project.
EXCLUDED_SPLITS_FROM_E3_TRAIN = frozenset({"test", "val"})
# Game challenges may use train+val; E3 *training* may only use eligible external /
# discovery / approved-train-frame contributions that pass all gates — never test/val GT.
ALLOWED_GAME_SPLITS = frozenset({"train", "val"})


@dataclass
class EligibilityDecision:
    eligible_for_e3_training: bool
    reasons: list[str] = field(default_factory=list)
    scene_group: str | None = None
    origin_split: str | None = None
    frame_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_manifest(manifest_path: Path | None = None) -> pd.DataFrame:
    path = manifest_path or (ROOT / "data" / "processed" / "split_manifest.csv")
    if not path.is_file():
        raise FileNotFoundError(f"split_manifest.csv introuvable: {path}")
    return pd.read_csv(path)


def locked_test_frame_ids(manifest_path: Path | None = None) -> set[str]:
    df = _load_manifest(manifest_path)
    return set(df.loc[df["split"].astype(str).str.lower() == "test", "frame_id"].astype(str))


def locked_val_frame_ids(manifest_path: Path | None = None) -> set[str]:
    df = _load_manifest(manifest_path)
    return set(df.loc[df["split"].astype(str).str.lower() == "val", "frame_id"].astype(str))


def frame_split(frame_id: str | None, manifest_path: Path | None = None) -> str | None:
    if not frame_id:
        return None
    df = _load_manifest(manifest_path)
    hit = df.loc[df["frame_id"].astype(str) == str(frame_id)]
    if hit.empty:
        return None
    return str(hit.iloc[0]["split"]).lower()


def frame_scene_group(frame_id: str | None, manifest_path: Path | None = None) -> str | None:
    if not frame_id:
        return None
    df = _load_manifest(manifest_path)
    hit = df.loc[df["frame_id"].astype(str) == str(frame_id)]
    if hit.empty:
        return str(frame_id)
    if "scene_group" in hit.columns:
        return str(hit.iloc[0]["scene_group"])
    return str(frame_id)


def near_duplicate_fingerprint(image_rgb: np.ndarray, *, size: int = 16) -> str:
    """Coarse fingerprint for near-duplicate detection (not cryptographic)."""
    arr = np.asarray(image_rgb)
    if arr.ndim == 2:
        gray = arr
    else:
        gray = arr.mean(axis=2)
    small = np.asarray(
        Image.fromarray(gray.astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    flat = small.ravel()
    mean = float(flat.mean())
    bits = (flat >= mean).astype(np.uint8)
    # Pack to hex
    as_bytes = np.packbits(bits)
    return as_bytes.tobytes().hex()


def hamming_hex(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 10**9
    ba = bytes.fromhex(a)
    bb = bytes.fromhex(b)
    return sum(bin(x ^ y).count("1") for x, y in zip(ba, bb))


def assess_contribution_eligibility(
    meta: dict[str, Any],
    *,
    manifest_path: Path | None = None,
    require_consent: bool = True,
    require_image_retained: bool = True,
    require_source_and_license: bool = True,
) -> EligibilityDecision:
    """Decide whether a contribution may enter an E3 *training* export.

    Distinguishes:
    - submitted / reviewed / approved  (review workflow)
    - eligible_for_e3_training         (stricter, this function)
    """
    reasons: list[str] = []
    frame_id = meta.get("frame_id")
    origin = frame_split(frame_id, manifest_path) if frame_id else None
    scene = frame_scene_group(frame_id, manifest_path) if frame_id else (
        f"sha:{str(meta.get('image_sha256') or '')[:16]}" if meta.get("image_sha256") else None
    )

    status = str(meta.get("verification_status") or "")
    if status != "approved":
        reasons.append(f"status_not_approved:{status or 'missing'}")

    if require_consent and not meta.get("consent_to_contribute"):
        reasons.append("missing_consent")

    if require_image_retained and not meta.get("image_retained"):
        reasons.append("image_not_retained")

    if require_source_and_license:
        src = str(meta.get("image_source") or "").strip()
        lic = str(meta.get("license_info") or "").strip()
        if len(src) < 3:
            reasons.append("missing_or_weak_source")
        if len(lic) < 3:
            reasons.append("missing_or_weak_license")

    if meta.get("duplicate_of") or status == "duplicate":
        reasons.append("marked_duplicate")

    if origin in EXCLUDED_SPLITS_FROM_E3_TRAIN:
        reasons.append(f"origin_split_excluded_from_e3_train:{origin}")

    # Explicitly block any contribution whose frame is in locked test/val sets
    if frame_id:
        if str(frame_id) in locked_test_frame_ids(manifest_path):
            reasons.append("locked_test_frame")
        if str(frame_id) in locked_val_frame_ids(manifest_path):
            reasons.append("locked_val_frame")

    # Training-challenge frames from the *train* split may be eligible only if approved
    # and consented; still reported with origin_split=train for leakage audits.
    challenge_type = str(meta.get("challenge_type") or "")
    if challenge_type == "training" and origin == "train":
        # Allowed in principle if other gates pass — no extra reject.
        pass
    elif challenge_type == "training" and origin != "train":
        reasons.append(f"training_challenge_non_train_origin:{origin}")

    eligible = len(reasons) == 0
    if eligible:
        reasons.append("eligible")

    return EligibilityDecision(
        eligible_for_e3_training=eligible,
        reasons=reasons,
        scene_group=scene,
        origin_split=origin,
        frame_id=str(frame_id) if frame_id else None,
    )


def detect_annotation_conflicts(
    contributions: list[dict[str, Any]],
    *,
    store_load_mask,
    iou_conflict_below: float = 0.5,
) -> list[dict[str, Any]]:
    """Find same-image contributions with low IoU masks → needs additional review.

    Does not claim automatic consensus. Returns conflict records only.
    """
    from metrics import binary_segmentation_metrics

    by_sha: dict[str, list[dict[str, Any]]] = {}
    for row in contributions:
        sha = row.get("image_sha256")
        if not sha:
            continue
        by_sha.setdefault(str(sha), []).append(row)

    conflicts: list[dict[str, Any]] = []
    for sha, rows in by_sha.items():
        if len(rows) < 2:
            continue
        masks = []
        for row in rows:
            try:
                assets = store_load_mask(row["contribution_id"])
                m = assets.get("player_mask")
                if m is None:
                    continue
                masks.append((row["contribution_id"], (np.asarray(m) > 0).astype(np.uint8)))
            except Exception:
                continue
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                a_id, a = masks[i]
                b_id, b = masks[j]
                if a.shape != b.shape:
                    conflicts.append(
                        {
                            "image_sha256": sha,
                            "a": a_id,
                            "b": b_id,
                            "reason": "shape_mismatch",
                        }
                    )
                    continue
                iou = float(binary_segmentation_metrics(a, b)["iou"])
                if iou < iou_conflict_below:
                    conflicts.append(
                        {
                            "image_sha256": sha,
                            "a": a_id,
                            "b": b_id,
                            "iou": iou,
                            "reason": "low_iou_conflict",
                        }
                    )
    return conflicts
