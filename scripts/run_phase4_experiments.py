"""Phase 4 orchestration script (E2a / E2b).

SAFETY:
- Default mode is smoke-only: architecture checks with weights=None.
- It does NOT download ImageNet weights.
- It does NOT start training.
- It does NOT access the test set.

Authorized training (future / reviewed Colab run):
  python scripts/run_phase4_experiments.py \\
      --authorize-training \\
      --authorize-imagenet-download

Both authorization flags are required together.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from phase4_training import (  # noqa: E402
    assert_required_git_ancestry,
    build_phase4_model_for_experiment,
    load_phase4_config,
    make_phase4_metadata,
    phase4_artefact_dirs,
    train_e2a,
    train_e2b,
)


def smoke_architecture() -> dict:
    p4 = load_phase4_config()
    height = int(p4["input"]["height"])
    width = int(p4["input"]["width"])
    report = {"weights": None, "training_started": False, "imagenet_downloaded": False}

    for experiment in ("E2a", "E2b"):
        model, info = build_phase4_model_for_experiment(
            experiment,
            p4,
            weights=None,
            authorize_imagenet_download=False,
        )
        x = np.zeros((1, height, width, 3), dtype=np.float32)
        y = model.predict(x, verbose=0)
        assert y.shape == (1, height, width, 1), y.shape
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
        report[experiment] = {
            "output_shape": list(y.shape),
            "finetune": info.get("finetune"),
            "trainable_weights": int(len(model.trainable_weights)),
        }
        print(f"OK {experiment} smoke forward pass → {y.shape}")

    dirs = phase4_artefact_dirs(p4)
    meta = make_phase4_metadata(
        experiment_id="E2a_smoke_architecture",
        experiment="E2a",
        model_name="unet_mobilenetv2",
        architecture="U-Net + MobileNetV2 (weights=None smoke)",
        hyperparameters={"batch_size": p4["batch_size"], "lr": p4["E2a"]["learning_rate"]},
        p4=p4,
    )
    report["metadata_keys"] = sorted(meta.keys())
    report["artefact_dirs"] = {k: str(v) for k, v in dirs.items() if k != "never_write_to"}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorize-training",
        action="store_true",
        help="Required (with --authorize-imagenet-download) to start real E2a/E2b training.",
    )
    parser.add_argument(
        "--authorize-imagenet-download",
        action="store_true",
        help="Required (with --authorize-training) to download ImageNet MobileNetV2 weights.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help=(
            "Explicit smoke mode: architecture checks with weights=None. "
            "This is also the default when no authorization flags are provided. "
            "Cannot be combined with authorization flags."
        ),
    )
    args = parser.parse_args()

    auth_train = bool(args.authorize_training)
    auth_imagenet = bool(args.authorize_imagenet_download)
    explicit_smoke = bool(args.smoke_only)
    want_train = auth_train or auth_imagenet

    print("Phase 4 script starting.")
    print("  authorize_training =", auth_train)
    print("  authorize_imagenet_download =", auth_imagenet)
    print("  smoke_only =", explicit_smoke or not want_train)

    if explicit_smoke and want_train:
        print(
            "ERROR: --smoke-only cannot be combined with authorization flags.",
            file=sys.stderr,
        )
        return 2

    if want_train and not (auth_train and auth_imagenet):
        print(
            "ERROR: both --authorize-training and --authorize-imagenet-download "
            "are required together before training or ImageNet download.",
            file=sys.stderr,
        )
        return 2

    if not want_train:
        # Default / explicit smoke path — never downloads ImageNet, never trains.
        report = smoke_architecture()
        print("Smoke report:", report)
        print("No training started. No ImageNet download.")
        return 0

    # Authorized training path — ancestry check before any download/fit.
    ancestry = assert_required_git_ancestry()
    print("Git ancestry OK:", ancestry)

    e2a = train_e2a(
        authorize_training=True,
        authorize_imagenet_download=True,
        weights="imagenet",
    )
    print("E2a checkpoint:", e2a["checkpoint_path"])
    print("E2a selected val threshold:", e2a["metadata"]["threshold_selection"]["value"])

    e2b = train_e2b(
        e2a_checkpoint=e2a["checkpoint_path"],
        authorize_training=True,
        authorize_imagenet_download=True,
    )
    print("E2b checkpoint:", e2b["checkpoint_path"])
    print("E2b selected val threshold:", e2b["metadata"]["threshold_selection"]["value"])
    print("Phase 4 training finished. Test set was not used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
