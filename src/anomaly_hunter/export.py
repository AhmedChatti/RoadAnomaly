"""Export *eligible* annotations into a versioned candidate dataset (E3 prep only).

Does NOT train models. Does NOT modify E2b. Does NOT touch the locked test split.
Approval alone is insufficient — see eligibility.assess_contribution_eligibility.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anomaly_hunter.eligibility import assess_contribution_eligibility
from anomaly_hunter.store import AnnotationStore, ContributionStatus

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_ROOT = ROOT / "data" / "annotations" / "exports"


def _unique_version_dir(export_root: Path, stamp: str) -> Path:
    """Never overwrite an existing dataset version directory."""
    base = export_root / f"v{stamp}"
    if not base.exists():
        return base
    for i in range(1, 1000):
        candidate = export_root / f"v{stamp}_{i:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Impossible de créer une version d’export unique.")


def export_approved_dataset(
    store: AnnotationStore,
    *,
    export_root: Path | None = None,
    require_image: bool = True,
    admin_token: str | None = None,
    enforce_admin: bool = False,
) -> dict[str, Any]:
    """Export contributions that are approved *and* E3-training-eligible."""
    if enforce_admin:
        from anomaly_hunter.admin_auth import require_admin

        require_admin(admin_token)

    export_root = Path(export_root) if export_root else DEFAULT_EXPORT_ROOT
    export_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Tiny sleep avoidance: if two exports same second, uniquify
    version_dir = _unique_version_dir(export_root, stamp)
    if version_dir.exists():
        raise RuntimeError(f"Refus d’écraser l’export existant: {version_dir}")
    images_dir = version_dir / "images"
    masks_dir = version_dir / "masks"
    meta_dir = version_dir / "meta"
    for d in (images_dir, masks_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    approved = store.list_contributions(status=ContributionStatus.APPROVED.value)
    seen_sha: set[str] = set()
    seen_near: list[str] = []
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    skipped_dup = 0
    skipped_near = 0
    skipped_no_image = 0

    for row in approved:
        decision = assess_contribution_eligibility(row)
        if not decision.eligible_for_e3_training:
            excluded.append(
                {
                    "contribution_id": row.get("contribution_id"),
                    "reasons": decision.reasons,
                    "frame_id": decision.frame_id,
                    "origin_split": decision.origin_split,
                }
            )
            continue

        sha = row.get("image_sha256") or ""
        if sha and sha in seen_sha:
            skipped_dup += 1
            excluded.append(
                {
                    "contribution_id": row.get("contribution_id"),
                    "reasons": ["duplicate_sha_in_export"],
                }
            )
            continue

        near = row.get("near_duplicate_fp")
        if near:
            from anomaly_hunter.eligibility import hamming_hex

            if any(hamming_hex(near, prev) <= 5 for prev in seen_near):
                skipped_near += 1
                excluded.append(
                    {
                        "contribution_id": row.get("contribution_id"),
                        "reasons": ["near_duplicate_in_export"],
                    }
                )
                continue

        assets = store.load_contribution_assets(row["contribution_id"])
        cdir = store.contributions_dir / row["contribution_id"]
        # Path traversal guard: resolved path must stay under contributions_dir
        resolved = cdir.resolve()
        if not str(resolved).startswith(str(store.contributions_dir.resolve())):
            excluded.append(
                {
                    "contribution_id": row.get("contribution_id"),
                    "reasons": ["path_traversal_blocked"],
                }
            )
            continue

        img_src = cdir / "image.png"
        mask_src = cdir / "player_mask.png"
        if not mask_src.is_file():
            excluded.append(
                {
                    "contribution_id": row.get("contribution_id"),
                    "reasons": ["missing_player_mask"],
                }
            )
            continue
        if require_image and not img_src.is_file():
            skipped_no_image += 1
            excluded.append(
                {
                    "contribution_id": row.get("contribution_id"),
                    "reasons": ["missing_retained_image"],
                }
            )
            continue

        cid = row["contribution_id"]
        if img_src.is_file():
            shutil.copy2(img_src, images_dir / f"{cid}.png")
        shutil.copy2(mask_src, masks_dir / f"{cid}.png")
        export_meta = {
            **assets["meta"],
            "eligibility": decision.to_dict(),
            "export_role": "e3_training_candidate_only",
            "not_stage5_data": True,
        }
        (meta_dir / f"{cid}.json").write_text(
            json.dumps(export_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if sha:
            seen_sha.add(sha)
        if near:
            seen_near.append(near)
        kept.append(
            {
                "contribution_id": cid,
                "frame_id": row.get("frame_id"),
                "image_sha256": sha,
                "near_duplicate_fp": near,
                "challenge_type": row.get("challenge_type"),
                "image_source": row.get("image_source"),
                "license_info": row.get("license_info"),
                "scene_group": decision.scene_group,
                "origin_split": decision.origin_split,
                "eligible_for_e3_training": True,
            }
        )

    # Scene-group leakage advisory for future split planning (no auto-split here)
    scene_groups = sorted({k["scene_group"] for k in kept if k.get("scene_group")})

    manifest = {
        "dataset_version": version_dir.name,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "n_eligible_exported": len(kept),
        "n_excluded": len(excluded),
        "skipped_duplicates": skipped_dup,
        "skipped_near_duplicates": skipped_near,
        "skipped_missing_image": skipped_no_image,
        "scene_groups": scene_groups,
        "scientific_note": (
            "Eligible human annotations only. Not merged into Stage 0–5 data. "
            "Locked test and validation frames are excluded from E3 training exports. "
            "E2b remains the immutable reference. Future E3 training is manual/documented."
        ),
        "forbidden": [
            "locked_test_split",
            "locked_val_split_for_e3_train",
            "automatic_e2b_retrain",
            "auto_promote",
            "overwrite_existing_version",
        ],
        "eligibility_policy": "anomaly_hunter.eligibility.assess_contribution_eligibility",
        "items": kept,
        "excluded": excluded,
    }
    (version_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Ensure we did not clobber: write a lock marker
    (version_dir / "EXPORT_COMPLETE").write_text(
        f"completed_at={manifest['created_utc']}\n", encoding="utf-8"
    )
    return manifest
