"""Create a verified local backup of the canonical Phase 3 E1 checkpoint.

Does not overwrite the original. Does not upload to Drive (manual step).
Does not start training.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reproducibility import EXPECTED_PHASE3_CHECKPOINT_SHA256, sha256_file  # noqa: E402

CANONICAL = "E1_cnn_scratch_20261108_1510"
SOURCE = ROOT / "checkpoints" / CANONICAL / "best.keras"
DEST_DIR = ROOT / "backups" / "phase3" / CANONICAL


def main() -> int:
    if not SOURCE.is_file():
        print(f"FAIL missing source checkpoint: {SOURCE}")
        return 1

    source_sha = sha256_file(SOURCE)
    print(f"Source: {SOURCE}")
    print(f"SHA-256: {source_sha}")
    if source_sha != EXPECTED_PHASE3_CHECKPOINT_SHA256:
        print("FAIL SHA-256 does not match the frozen Phase 3 value.")
        print(f"Expected: {EXPECTED_PHASE3_CHECKPOINT_SHA256}")
        return 1
    print("OK matches frozen Phase 3 SHA-256")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    dest = DEST_DIR / "best.keras"
    if dest.exists():
        existing = sha256_file(dest)
        if existing == source_sha:
            print(f"OK backup already present and identical: {dest}")
        else:
            # Never overwrite a conflicting backup silently.
            stamped = DEST_DIR / f"best_conflict_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.keras"
            print(f"WARN existing backup has different hash; leaving it and writing {stamped.name}")
            shutil.copy2(SOURCE, stamped)
            dest = stamped
    else:
        shutil.copy2(SOURCE, dest)
        print(f"Wrote {dest}")

    backup_sha = sha256_file(dest)
    if backup_sha != source_sha:
        print("FAIL backup hash mismatch after copy")
        return 1

    freeze_src = SOURCE.parent / "FREEZE.txt"
    if freeze_src.is_file():
        shutil.copy2(freeze_src, DEST_DIR / "FREEZE.txt")

    # Lightweight sidecar copies (metadata JSON is also under experiments/).
    meta_src = ROOT / "experiments" / "metadata" / f"{CANONICAL}.json"
    if meta_src.is_file():
        shutil.copy2(meta_src, DEST_DIR / f"{CANONICAL}.json")

    manifest = {
        "canonical_experiment_id": CANONICAL,
        "source_path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "backup_path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "sha256": backup_sha,
        "verified_against_freeze": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "drive_upload_targets": [
            f"My Drive/RoadAnomaly/checkpoints/phase3/{CANONICAL}/best.keras",
            f"My Drive/RoadAnomaly/backups/phase3/{CANONICAL}/best.keras",
        ],
        "note": "Original local checkpoint was not modified. Upload to Drive is manual.",
    }
    manifest_path = DEST_DIR / "BACKUP_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (DEST_DIR / "SHA256.txt").write_text(backup_sha + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
