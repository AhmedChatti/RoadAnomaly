"""Final acceptance audit tests — security, eligibility, freeze, workflows."""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXPECTED_E2B_SHA = "280170dc64347b0b4cc8ce3af9be40dec4ac3733b27d1abeb620cc0c8e88e432"

# Pin Stage 5 / freeze artifact content hashes observed at audit start.
FROZEN_ARTIFACT_SHA256 = {
    "experiments/metadata/phase4/FINAL_MODEL_SELECTION_FREEZE.json": (
        "1f75e04e5b4ecf2a1cf8353269b201fd8410dc1d538b630be26e73fb7656f99f"
    ),
    "experiments/metadata/phase4/STAGE5_TEST_EVAL_DONE.json": (
        "722652c697c97c9d9ce386adff223fe5bdebe87224c16f66f2a3ac8ac53958cb"
    ),
    "experiments/results/phase4/phase4_final_test_comparison.csv": (
        "826a3cf5d7d9ea5170df8776b3d00b82e2632d5c1592359ee89232f4004b25e6"
    ),
}


class FreezeIntegrityTests(unittest.TestCase):
    def test_e2b_constants(self):
        from app_inference import E2B_CHECKPOINT_SHA256, E2B_THRESHOLD

        self.assertEqual(E2B_THRESHOLD, 0.55)
        self.assertEqual(E2B_CHECKPOINT_SHA256, EXPECTED_E2B_SHA)

    def test_checkpoint_file_sha_if_present(self):
        ckpt = ROOT / "checkpoints/phase4/E2b_unet_mobilenetv2_finetune_20261108_1721/best.keras"
        if not ckpt.is_file():
            self.skipTest("checkpoint not on disk")
        h = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        self.assertEqual(h, EXPECTED_E2B_SHA)

    def test_stage5_artifacts_unchanged(self):
        for rel, expected in FROZEN_ARTIFACT_SHA256.items():
            path = ROOT / rel
            self.assertTrue(path.is_file(), msg=f"missing {rel}")
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(h, expected, msg=f"FROZEN ARTIFACT CHANGED: {rel}")

    def test_stage5_e2b_row(self):
        import pandas as pd

        df = pd.read_csv(ROOT / "experiments/results/phase4/phase4_final_test_comparison.csv")
        row = df.loc[df["experiment_id"].astype(str).str.startswith("E2b")].iloc[0]
        self.assertAlmostEqual(float(row["threshold"]), 0.55)
        self.assertEqual(str(row["checkpoint_sha256"]), EXPECTED_E2B_SHA)
        self.assertGreater(float(row["mean_dice"]), 0.8)


class AdminAuthTests(unittest.TestCase):
    def test_missing_token_fails_closed(self):
        from anomaly_hunter.admin_auth import admin_auth_status, verify_admin_token

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROADANOMALY_ADMIN_TOKEN", None)
            self.assertFalse(admin_auth_status().ok)
            self.assertFalse(verify_admin_token("anything").ok)

    def test_banned_default_rejected(self):
        from anomaly_hunter.admin_auth import admin_auth_status, verify_admin_token

        with mock.patch.dict(
            os.environ, {"ROADANOMALY_ADMIN_TOKEN": "roadanomaly-local-admin"}
        ):
            self.assertFalse(admin_auth_status().ok)
            self.assertFalse(verify_admin_token("roadanomaly-local-admin").ok)

    def test_short_token_rejected(self):
        from anomaly_hunter.admin_auth import admin_auth_status

        with mock.patch.dict(os.environ, {"ROADANOMALY_ADMIN_TOKEN": "short"}):
            self.assertFalse(admin_auth_status().ok)

    def test_strong_token_accepted(self):
        from anomaly_hunter.admin_auth import require_admin, verify_admin_token

        secret = "unit-test-admin-token-ok"
        with mock.patch.dict(os.environ, {"ROADANOMALY_ADMIN_TOKEN": secret}):
            self.assertTrue(verify_admin_token(secret).ok)
            self.assertFalse(verify_admin_token("wrong-token-xxxxxx").ok)
            with self.assertRaises(PermissionError):
                require_admin("wrong-token-xxxxxx")
            require_admin(secret)


class EligibilityAndExportTests(unittest.TestCase):
    def test_test_and_val_frames_excluded(self):
        import pandas as pd

        from anomaly_hunter.eligibility import assess_contribution_eligibility

        df = pd.read_csv(ROOT / "data/processed/split_manifest.csv")
        test_id = str(df.loc[df["split"] == "test", "frame_id"].iloc[0])
        val_id = str(df.loc[df["split"] == "val", "frame_id"].iloc[0])
        train_id = str(df.loc[df["split"] == "train", "frame_id"].iloc[0])

        for fid, expect_ok in ((test_id, False), (val_id, False)):
            meta = {
                "verification_status": "approved",
                "consent_to_contribute": True,
                "image_retained": True,
                "image_source": "RoadAnomaly research",
                "license_info": "CC BY-SA 4.0",
                "frame_id": fid,
                "challenge_type": "training",
            }
            d = assess_contribution_eligibility(meta)
            self.assertFalse(d.eligible_for_e3_training, msg=fid)
            self.assertFalse(expect_ok and d.eligible_for_e3_training)

        meta_train = {
            "verification_status": "approved",
            "consent_to_contribute": True,
            "image_retained": True,
            "image_source": "RoadAnomaly research",
            "license_info": "CC BY-SA 4.0",
            "frame_id": train_id,
            "challenge_type": "training",
        }
        self.assertTrue(assess_contribution_eligibility(meta_train).eligible_for_e3_training)

        # Approval alone insufficient
        weak = dict(meta_train)
        weak["consent_to_contribute"] = False
        self.assertFalse(assess_contribution_eligibility(weak).eligible_for_e3_training)

    def test_pool_never_serves_test(self):
        from anomaly_hunter.pool import FORBIDDEN_SPLITS, load_training_challenges

        for c in load_training_challenges():
            self.assertNotIn(c.split, FORBIDDEN_SPLITS)

    def test_export_excludes_val_and_does_not_overwrite(self):
        from anomaly_hunter.export import export_approved_dataset
        from anomaly_hunter.store import AnnotationStore

        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(root=Path(tmp) / "ann")
            rgb = np.random.randint(0, 255, (32, 40, 3), dtype=np.uint8)
            mask = np.zeros((32, 40), dtype=np.uint8)
            mask[5:20, 5:25] = 1
            import pandas as pd

            train_id = str(
                pd.read_csv(ROOT / "data/processed/split_manifest.csv")
                .loc[lambda d: d["split"] == "train", "frame_id"]
                .iloc[0]
            )
            c = store.submit(
                challenge_type="discovery",
                player_session_id="s",
                consent_to_contribute=True,
                player_mask=mask,
                image_rgb=rgb,
                image_bytes=None,
                ai_mask=mask,
                ai_probability=None,
                image_source="photo personnelle test",
                license_info="CC0 unit test",
                frame_id=None,
                retain_image=True,
            )
            store.review(c.contribution_id, status="approved", comment="ok")

            # Also approve a fake val-frame contribution — must be excluded
            val_id = str(
                pd.read_csv(ROOT / "data/processed/split_manifest.csv")
                .loc[lambda d: d["split"] == "val", "frame_id"]
                .iloc[0]
            )
            c_val = store.submit(
                challenge_type="training",
                player_session_id="s2",
                consent_to_contribute=True,
                player_mask=mask,
                image_rgb=rgb,
                image_bytes=None,
                ai_mask=mask,
                ai_probability=None,
                image_source="RoadAnomaly research",
                license_info="CC BY-SA 4.0",
                frame_id=val_id,
                retain_image=True,
            )
            store.review(c_val.contribution_id, status="approved", comment="should exclude")

            export_root = Path(tmp) / "exports"
            m1 = export_approved_dataset(store, export_root=export_root)
            self.assertIn("n_eligible_exported", m1)
            self.assertGreaterEqual(m1["n_eligible_exported"], 1)
            # val excluded
            self.assertTrue(
                any("locked_val" in str(e.get("reasons")) or "origin_split_excluded" in str(e.get("reasons"))
                    for e in m1["excluded"])
            )
            # no overwrite
            vdir = export_root / m1["dataset_version"]
            self.assertTrue((vdir / "manifest.json").is_file())
            m2 = export_approved_dataset(store, export_root=export_root)
            self.assertNotEqual(m1["dataset_version"], m2["dataset_version"])

            # unauthorized export
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ROADANOMALY_ADMIN_TOKEN", None)
                with self.assertRaises(PermissionError):
                    export_approved_dataset(
                        store, export_root=export_root, enforce_admin=True, admin_token="x"
                    )


class AnalyserWorkflowTests(unittest.TestCase):
    def test_upload_formats_and_reject(self):
        from app_inference import load_user_image, preprocess_for_e2b, predict_e2b

        def png_bytes(size=(64, 80)):
            buf = io.BytesIO()
            Image.fromarray(np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8)).save(
                buf, format="PNG"
            )
            return buf.getvalue()

        def jpeg_bytes():
            buf = io.BytesIO()
            Image.fromarray(np.random.randint(0, 255, (50, 70, 3), dtype=np.uint8)).save(
                buf, format="JPEG"
            )
            return buf.getvalue()

        for name, data in (
            ("a.png", png_bytes()),
            ("a.jpg", jpeg_bytes()),
            ("a.jpeg", jpeg_bytes()),
        ):
            rgb = load_user_image(data, filename=name)
            self.assertEqual(rgb.ndim, 3)
            self.assertEqual(rgb.shape[2], 3)

        with self.assertRaises(ValueError):
            load_user_image(b"not-image", filename="x.gif")

        class Fake:
            def predict(self, batch, verbose=0):
                from app_inference import E2B_INPUT_HEIGHT, E2B_INPUT_WIDTH

                p = np.full((1, E2B_INPUT_HEIGHT, E2B_INPUT_WIDTH, 1), 0.6, dtype=np.float32)
                return p

        rgb = load_user_image(png_bytes((120, 160)), filename="t.png")
        result = predict_e2b(Fake(), rgb)
        self.assertEqual(result.threshold, 0.55)
        self.assertEqual(result.binary_full.shape, rgb.shape[:2])
        self.assertEqual(result.overlay_rgb.shape, rgb.shape)


class GameScoringAntiReplayTests(unittest.TestCase):
    def test_replay_does_not_inflate_points(self):
        from anomaly_hunter.gamification import apply_training_score, default_profile

        p = default_profile("sess")
        p = apply_training_score(
            p, challenge_id="train::a", points_earned=80, player_dice=0.8, beat_ai=False
        )
        self.assertEqual(p.points, 80)
        p = apply_training_score(
            p, challenge_id="train::a", points_earned=80, player_dice=0.8, beat_ai=False
        )
        self.assertEqual(p.points, 80)

    def test_empty_mask_rejected_by_store(self):
        from anomaly_hunter.store import AnnotationStore

        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(root=Path(tmp))
            empty = np.zeros((16, 16), dtype=np.uint8)
            with self.assertRaises(ValueError):
                store.submit(
                    challenge_type="discovery",
                    player_session_id="s",
                    consent_to_contribute=True,
                    player_mask=empty,
                    image_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
                    image_bytes=b"abc",
                    ai_mask=None,
                    ai_probability=None,
                    image_source="src",
                    license_info="lic",
                    retain_image=True,
                )


class PathTraversalTests(unittest.TestCase):
    def test_invalid_contribution_id(self):
        from anomaly_hunter.store import AnnotationStore

        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(root=Path(tmp))
            with self.assertRaises(ValueError):
                store._contrib_dir("../etc/passwd000000000000000000000")
            with self.assertRaises(ValueError):
                store._contrib_dir("not-hex")


if __name__ == "__main__":
    unittest.main()
