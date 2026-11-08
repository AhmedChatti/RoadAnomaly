"""Create the Google Drive RoadAnomaly folder skeleton.

Usage (after mounting Drive in Colab, or on a synced Drive path locally):

  python scripts/init_drive_layout.py --root "/content/drive/MyDrive/RoadAnomaly"

Does not copy dataset files or checkpoints. Does not start training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SUBDIRS = [
    "dataset/original",
    "checkpoints/phase3/E1_cnn_scratch_20261108_1510",
    "checkpoints/phase4/E2a",
    "checkpoints/phase4/E2b",
    "experiments/metadata",
    "experiments/results",
    "experiments/figures",
    "backups/phase3",
    "backups/phase4",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help='Drive project root, e.g. "/content/drive/MyDrive/RoadAnomaly"',
    )
    args = parser.parse_args()
    root = Path(args.root)
    created = []
    for rel in SUBDIRS:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))
    readme = root / "README_DRIVE.txt"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    "RoadAnomaly persistent Drive store",
                    "",
                    "dataset/original/  <- upload frames/ + keep frame_list.json, credits.txt, readme.md",
                    "  frames/*.webp",
                    "  frames/*.labels/labels_semantic.png  (and sibling LabelGrab files)",
                    "checkpoints/phase3/E1_cnn_scratch_20261108_1510/best.keras",
                    "checkpoints/phase4/E2a|E2b/",
                    "experiments/{metadata,results,figures}/",
                    "backups/",
                    "",
                    "Source code lives on GitHub; Colab clones/pulls it. Do not treat Drive as the code repo.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    print(f"Drive root: {root}")
    for path in created:
        print(f"  OK {path}")
    print("Done. Next: upload original dataset and Phase 3 checkpoint (see docs/11).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
