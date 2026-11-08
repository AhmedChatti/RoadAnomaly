"""Phase 4 training/eval safety tests — no real training, no ImageNet download."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from phase4_evaluation import evaluate_phase4_on_split  # noqa: E402
from phase4_training import (  # noqa: E402
    assert_not_writing_to_phase3,
    assert_required_git_ancestry,
    build_phase4_model_for_experiment,
    finalize_phase4_threshold_selection,
    load_phase4_config,
    make_phase4_metadata,
    phase4_artefact_dirs,
    threshold_grid_from_config,
    train_e2a,
    train_e2b,
)


class Phase4TrainingGateTests(unittest.TestCase):
    def test_train_e2a_blocked_without_authorization(self):
        with self.assertRaises(RuntimeError):
            train_e2a(authorize_training=False, authorize_imagenet_download=False)

    def test_train_e2a_blocked_when_only_one_flag(self):
        with self.assertRaises(RuntimeError) as ctx:
            train_e2a(authorize_training=True, authorize_imagenet_download=False, weights="imagenet")
        self.assertIn("together", str(ctx.exception).lower())

    def test_train_e2b_blocked_without_authorization(self):
        with self.assertRaises(RuntimeError):
            train_e2b(e2a_checkpoint="dummy.keras", authorize_training=False)

    def test_train_e2b_requires_existing_checkpoint(self):
        # Dual auth would also require git ancestry + imagenet; patch those away and
        # ensure missing E2a checkpoint still fails safely without training.
        with mock.patch("phase4_training._require_training_authorization"):
            with self.assertRaises(FileNotFoundError):
                train_e2b(
                    e2a_checkpoint=str(ROOT / "checkpoints" / "phase4" / "missing" / "best.keras"),
                    authorize_training=True,
                    authorize_imagenet_download=True,
                )

    def test_build_rejects_imagenet_without_flag(self):
        with self.assertRaises(RuntimeError):
            build_phase4_model_for_experiment(
                "E2a", weights="imagenet", authorize_imagenet_download=False
            )

    def test_metadata_contains_reproducibility_fields(self):
        meta = make_phase4_metadata(
            experiment_id="E2a_unit_meta",
            experiment="E2a",
            model_name="unet_mobilenetv2",
            architecture="U-Net + MobileNetV2",
            hyperparameters={"lr": 0.001},
        )
        for key in (
            "experiment_id",
            "created_at_utc",
            "git_commit_sha",
            "dataset_fingerprint",
            "split_manifest_sha256",
            "input_height",
            "input_width",
            "seed",
            "software_versions",
            "threshold_selection",
            "normalize",
        ):
            self.assertIn(key, meta)
        self.assertEqual(meta["threshold_selection"]["default_during_training"], 0.5)
        self.assertEqual(meta["normalize"], "mobilenet_v2")
        self.assertEqual(meta["status"], "initialized_no_training")

    def test_artefact_dirs_isolated_under_phase4(self):
        dirs = phase4_artefact_dirs()
        self.assertTrue(str(dirs["checkpoints"]).replace("\\", "/").endswith("checkpoints/phase4"))
        self.assertTrue(str(dirs["metadata"]).replace("\\", "/").endswith("metadata/phase4"))
        self.assertTrue(str(dirs["results"]).replace("\\", "/").endswith("results/phase4"))
        self.assertTrue(str(dirs["figures"]).replace("\\", "/").endswith("figures/phase4"))
        p4 = load_phase4_config()
        target = (ROOT / p4["artefacts"]["never_write_to"][0] / "best.keras").resolve()
        with self.assertRaises(RuntimeError):
            assert_not_writing_to_phase3(target, p4)

    def test_refuses_overwrite_phase3_metadata_file(self):
        p4 = load_phase4_config()
        e1_meta = ROOT / "experiments" / "metadata" / "E1_cnn_scratch_20261108_1510.json"
        if e1_meta.is_file():
            with self.assertRaises(RuntimeError):
                assert_not_writing_to_phase3(e1_meta, p4)


class Phase4GitAncestryTests(unittest.TestCase):
    def test_required_commit_is_ancestor_of_head(self):
        info = assert_required_git_ancestry()
        p4 = load_phase4_config()
        self.assertEqual(info["required"], p4["git_commit_required"])
        self.assertEqual(len(info["head"]), 40)
        self.assertEqual(info["ancestry"], "ok")

    def test_missing_required_commit_fails(self):
        p4 = load_phase4_config()
        bad = dict(p4)
        bad["git_commit_required"] = "deadbeef" * 5
        with self.assertRaises(RuntimeError) as ctx:
            assert_required_git_ancestry(bad)
        self.assertIn("not found", str(ctx.exception).lower())


class Phase4ThresholdIntegrationTests(unittest.TestCase):
    def test_threshold_grid_matches_config(self):
        grid = threshold_grid_from_config()
        p4 = load_phase4_config()
        self.assertEqual(len(grid), int(p4["threshold_selection"]["grid_steps"]))
        self.assertEqual(grid[0], float(p4["threshold_selection"]["grid_start"]))
        self.assertEqual(grid[-1], float(p4["threshold_selection"]["grid_stop"]))

    def test_finalize_threshold_selection_writes_csv_and_metadata(self):
        import phase4_training as p4t

        class DummyModel:
            def predict(self, batch, verbose=0):
                import numpy as np

                n = batch.shape[0]
                h, w = batch.shape[1], batch.shape[2]
                return np.full((n, h, w, 1), 0.2, dtype=np.float32)

        calls = {"splits": []}

        def fake_select(predict_fn, thresholds=None, normalize="mobilenet_v2", batch_size=4):
            # Prove validation-only helper is used (wrapper hardcodes val).
            calls["normalize"] = normalize
            calls["thresholds"] = list(thresholds or [])
            return {
                "best_threshold": 0.35,
                "best_val_mean_dice": 0.42,
                "grid": pd.DataFrame(
                    [{"threshold": 0.35, "val_mean_dice": 0.42, "val_mean_iou": 0.3}]
                ),
                "best_result": {
                    "mean_per_image": {"mean_dice": 0.42, "mean_iou": 0.3},
                    "global": {"dice": 0.42},
                    "split": "val",
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_dirs = {
                "checkpoints": tmp_path / "checkpoints" / "phase4",
                "metadata": tmp_path / "experiments" / "metadata" / "phase4",
                "results": tmp_path / "experiments" / "results" / "phase4",
                "figures": tmp_path / "experiments" / "figures" / "phase4",
                "repo_root": ROOT,
                "never_write_to": [],
            }
            for key in ("checkpoints", "metadata", "results", "figures"):
                fake_dirs[key].mkdir(parents=True, exist_ok=True)

            meta = make_phase4_metadata(
                experiment_id="E2a_thr_unit",
                experiment="E2a",
                model_name="unet_mobilenetv2",
                architecture="unit",
                hyperparameters={},
            )
            meta["status"] = "completed"
            meta["threshold_selection"]["value"] = None

            with mock.patch.object(p4t, "phase4_artefact_dirs", return_value=fake_dirs):
                with mock.patch.object(p4t, "assert_not_writing_to_phase3", return_value=None):
                    with mock.patch(
                        "phase4_evaluation.select_phase4_threshold_on_validation",
                        side_effect=fake_select,
                    ):
                        out = finalize_phase4_threshold_selection(
                            model=DummyModel(),
                            experiment_id="E2a_thr_unit",
                            metadata=meta,
                        )

            self.assertEqual(out["threshold_selection"]["value"], 0.35)
            self.assertEqual(out["threshold_selection"]["default_during_training"], 0.5)
            self.assertEqual(out["threshold_selection"]["split"], "val")
            self.assertIsNotNone(out["threshold_selection"]["value"])
            csv_path = fake_dirs["results"] / "E2a_thr_unit_val_threshold_grid.csv"
            self.assertTrue(csv_path.is_file())
            self.assertEqual(calls["normalize"], "mobilenet_v2")
            self.assertEqual(out["status"], "completed_with_val_threshold")

    def test_finalize_failure_marks_incomplete_without_fabricated_threshold(self):
        import phase4_training as p4t

        class DummyModel:
            def predict(self, batch, verbose=0):
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_dirs = {
                "checkpoints": tmp_path / "checkpoints" / "phase4",
                "metadata": tmp_path / "experiments" / "metadata" / "phase4",
                "results": tmp_path / "experiments" / "results" / "phase4",
                "figures": tmp_path / "experiments" / "figures" / "phase4",
                "repo_root": ROOT,
                "never_write_to": [],
            }
            for key in ("checkpoints", "metadata", "results", "figures"):
                fake_dirs[key].mkdir(parents=True, exist_ok=True)
            meta = make_phase4_metadata(
                experiment_id="E2a_thr_fail",
                experiment="E2a",
                model_name="unet_mobilenetv2",
                architecture="unit",
                hyperparameters={},
            )
            with mock.patch.object(p4t, "phase4_artefact_dirs", return_value=fake_dirs):
                with mock.patch.object(p4t, "assert_not_writing_to_phase3", return_value=None):
                    with mock.patch(
                        "phase4_evaluation.select_phase4_threshold_on_validation",
                        side_effect=RuntimeError("val select failed"),
                    ):
                        with self.assertRaises(RuntimeError):
                            finalize_phase4_threshold_selection(
                                model=DummyModel(),
                                experiment_id="E2a_thr_fail",
                                metadata=meta,
                            )
            self.assertEqual(meta["status"], "incomplete_threshold_selection_failed")
            self.assertIsNone(meta["threshold_selection"]["value"])


class Phase4EvaluationSafetyTests(unittest.TestCase):
    def test_test_split_hard_blocked(self):
        def fake_predict(x):
            raise AssertionError("predict should not be called for test split")

        with self.assertRaises(RuntimeError):
            evaluate_phase4_on_split(fake_predict, "test", threshold=0.5)

    def test_threshold_selection_uses_validation_only(self):
        import evaluation as ev

        calls = []

        def fake_eval(predict_fn, split, **kwargs):
            calls.append(split)
            return {
                "mean_per_image": {"mean_dice": 0.1, "mean_iou": 0.05},
                "global": {},
                "per_image": pd.DataFrame(
                    [
                        {
                            "frame_id": "x",
                            "precision": 0,
                            "recall": 0,
                            "f1": 0,
                            "iou": 0,
                            "dice": 0.1,
                            "accuracy": 0,
                        }
                    ]
                ),
                "split": split,
                "threshold": kwargs.get("threshold", 0.5),
                "n_images": 1,
            }

        with mock.patch.object(ev, "evaluate_predictor_on_split", side_effect=fake_eval):
            out = ev.select_threshold_on_validation(
                lambda x: x,
                thresholds=[0.4, 0.5],
                normalize="mobilenet_v2",
            )
        self.assertEqual(set(calls), {"val"})
        self.assertIn("best_threshold", out)

    def test_select_phase4_threshold_wrapper_defaults_normalize(self):
        import phase4_evaluation

        seen = {}

        def fake_select(predict_fn, thresholds=None, normalize="scale_0_1", batch_size=4):
            seen["normalize"] = normalize
            return {
                "best_threshold": 0.5,
                "best_val_mean_dice": 0.0,
                "grid": pd.DataFrame(),
                "best_result": {},
            }

        with mock.patch.object(phase4_evaluation, "select_threshold_on_validation", side_effect=fake_select):
            phase4_evaluation.select_phase4_threshold_on_validation(lambda x: x)
        self.assertEqual(seen["normalize"], "mobilenet_v2")


class Phase4CheckpointMetricConventionTests(unittest.TestCase):
    def test_config_uses_val_mean_dice(self):
        p4 = load_phase4_config()
        self.assertEqual(p4["checkpoint_metric"], "val_mean_dice")
        self.assertEqual(p4["checkpoint_mode"], "max")
        self.assertEqual(p4["threshold_selection"]["split"], "val")
        self.assertEqual(p4["training_threshold"], 0.5)
        self.assertEqual(p4["batch_size"], 4)

    def test_config_hyperparameters_match_protocol(self):
        p4 = load_phase4_config()
        self.assertEqual(p4["E2a"]["learning_rate"], 0.001)
        self.assertEqual(p4["E2b"]["learning_rate"], 0.0001)
        self.assertEqual(p4["E2b"]["finetune_from_layer"], "block_13_expand")
        self.assertEqual(p4["input"]["normalize"], "mobilenet_v2")
        self.assertEqual(p4["decoder"]["final_conv_filters"], 16)
        self.assertFalse(p4["safety"]["allow_test_access"])


class Phase4SmokeScriptTests(unittest.TestCase):
    def test_default_cli_is_smoke_and_never_calls_train(self):
        import run_phase4_experiments as script

        with mock.patch.object(sys, "argv", ["run_phase4_experiments.py"]):
            with mock.patch.object(script, "smoke_architecture", return_value={"weights": None}) as smoke:
                with mock.patch.object(script, "train_e2a") as train_a:
                    with mock.patch.object(script, "train_e2b") as train_b:
                        code = script.main()
        self.assertEqual(code, 0)
        smoke.assert_called_once()
        train_a.assert_not_called()
        train_b.assert_not_called()

    def test_smoke_only_rejects_auth_flags(self):
        import run_phase4_experiments as script

        with mock.patch.object(
            sys,
            "argv",
            [
                "run_phase4_experiments.py",
                "--smoke-only",
                "--authorize-training",
                "--authorize-imagenet-download",
            ],
        ):
            code = script.main()
        self.assertEqual(code, 2)

    def test_single_auth_flag_rejected(self):
        import run_phase4_experiments as script

        with mock.patch.object(
            sys,
            "argv",
            ["run_phase4_experiments.py", "--authorize-training"],
        ):
            code = script.main()
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
