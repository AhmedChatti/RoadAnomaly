"""Export approved annotations into a versioned candidate dataset (E3 prep only).

Does NOT train models. Does NOT modify E2b. Does NOT touch the locked test split.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anomaly_hunter.store import AnnotationStore, ContributionStatus

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_ROOT = ROOT / "data" / "annotations" / "exports"


def export_approved_dataset(
    store: AnnotationStore,
    *,
    export_root: Path | None = None,
    require_image: bool = True,
) -> dict[str, Any]:
    """Copy approved contributions into a timestamped versioned folder."""
    export_root = Path(export_root) if export_root else DEFAULT_EXPORT_ROOT
    export_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = export_root / f"v{stamp}"
    images_dir = version_dir / "images"
    masks_dir = version_dir / "masks"
    meta_dir = version_dir / "meta"
    for d in (images_dir, masks_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    approved = store.list_contributions(status=ContributionStatus.APPROVED.value)
    # Deduplicate by image_sha256 (keep first)
    seen_sha: set[str] = set()
    kept: list[dict[str, Any]] = []
    skipped_dup = 0
    skipped_no_image = 0

    for row in approved:
        sha = row.get("image_sha256") or ""
        if sha and sha in seen_sha:
            skipped_dup += 1
            continue
        assets = store.load_contribution_assets(row["contribution_id"])
        cdir = store.contributions_dir / row["contribution_id"]
        img_src = cdir / "image.png"
        mask_src = cdir / "player_mask.png"
        if not mask_src.is_file():
            continue
        if require_image and not img_src.is_file():
            skipped_no_image += 1
            continue
        cid = row["contribution_id"]
        if img_src.is_file():
            shutil.copy2(img_src, images_dir / f"{cid}.png")
        shutil.copy2(mask_src, masks_dir / f"{cid}.png")
        (meta_dir / f"{cid}.json").write_text(
            json.dumps(assets["meta"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if sha:
            seen_sha.add(sha)
        kept.append(
            {
                "contribution_id": cid,
                "frame_id": row.get("frame_id"),
                "image_sha256": sha,
                "challenge_type": row.get("challenge_type"),
            }
        )

    manifest = {
        "dataset_version": f"v{stamp}",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "n_approved_exported": len(kept),
        "skipped_duplicates": skipped_dup,
        "skipped_missing_image": skipped_no_image,
        "scientific_note": (
            "Candidate human annotations only. Not merged into Stage 0–5 data. "
            "E2b remains the immutable reference. Future E3 training is manual/documented."
        ),
        "forbidden": ["locked_test_split", "automatic_e2b_retrain", "auto_promote"],
        "items": kept,
    }
    (version_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
