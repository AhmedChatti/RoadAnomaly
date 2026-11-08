"""Tests for Anomaly Hunter game logic (no E2b weight mutation, no test-split use)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_hunter.export import export_approved_dataset  # noqa: E402
from anomaly_hunter.gamification import apply_training_score, default_profile  # noqa: E402
from anomaly_hunter.masks import (  # noqa: E402
    binary_mask_from_canvas_json,
    load_ground_truth_mask,
    remap_semantic_to_binary,
    validate_player_mask,
)
from anomaly_hunter.pool import (  # noqa: E402
    FORBIDDEN_SPLITS,
    assert_not_test_frame,
    load_training_challenges,
)
from anomaly_hunter.scoring import score_training_round  # noqa: E402
from anomaly_hunter.store import AnnotationStore, ContributionStatus  # noqa: E402
from app_inference import E2B_CHECKPOINT_SHA256, E2B_THRESHOLD  # noqa: E402


class MaskTests(unittest.TestCase):
    def test_remap_semantic(self):
        m = np.array([[0, 2], [2, 0]], dtype=np.uint8)
        b = remap_semantic_to_binary(m)
        np.testing.assert_array_equal(b, np.array([[0, 1], [1, 0]], dtype=np.uint8))

    def test_validate_shape(self):
        mask = np.zeros((10, 12), dtype=np.uint8)
        out = validate_player_mask(mask, expected_hw=(10, 12))
        self.assertEqual(out.shape, (10, 12))
        with self.assertRaises(ValueError):
            validate_player_mask(mask, expected_hw=(8, 8))

    def test_canvas_json_to_mask_alignment(self):
        # Diagonal stroke on 40x60 canvas → 20x30 image
        objects = [
            {
                "type": "freedraw",
                "stroke": "#e85d2a",
                "strokeWidth": 8,
                "path": [["M", 5, 5], ["L", 35, 25]],
            }
        ]
        mask = binary_mask_from_canvas_json(
            {"objects": objects},
            image_hw=(20, 30),
            canvas_width=40,
            canvas_height=60,
        )
        self.assertEqual(mask.shape, (20, 30))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(int(mask.sum()), 0)

    def test_eraser_stroke(self):
        objects = [
            {
                "type": "freedraw",
                "stroke": "#e85d2a",
                "strokeWidth": 20,
                "path": [["M", 10, 10], ["L", 50, 10], ["L", 50, 50], ["L", 10, 50]],
            },
            {
                "type": "eraser",
                "stroke": "#ffffff",
                "strokeWidth": 12,
                "path": [["M", 20, 20], ["L", 40, 40]],
            },
        ]
        mask = binary_mask_from_canvas_json(
            {"objects": objects},
            image_hw=(60, 60),
            canvas_width=60,
            canvas_height=60,
        )
        self.assertEqual(mask.shape, (60, 60))


class PoolTests(unittest.TestCase):
    def test_never_includes_test_split(self):
        challenges = load_training_challenges()
        for c in challenges:
            self.assertNotIn(c.split, FORBIDDEN_SPLITS)
            self.assertIn(c.split, {"train", "val"})
        # Manifest-level guard
        import pandas as pd

        df = pd.read_csv(ROOT / "data/processed/split_manifest.csv")
        test_ids = set(df.loc[df["split"] == "test", "frame_id"].astype(str))
        for c in challenges:
            self.assertNotIn(c.frame_id, test_ids)

    def test_assert_not_test_frame(self):
        import pandas as pd

        df = pd.read_csv(ROOT / "data/processed/split_manifest.csv")
        test_id = str(df.loc[df["split"] == "test", "frame_id"].iloc[0])
        with self.assertRaises(PermissionError):
            assert_not_test_frame(test_id)
        train_id = str(df.loc[df["split"] == "train", "frame_id"].iloc[0])
        assert_not_test_frame(train_id)  # must not raise


class ScoringTests(unittest.TestCase):
    def test_perfect_overlap(self):
        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[8:24, 8:24] = 1
        score = score_training_round(ground_truth=gt, player_mask=gt.copy(), ai_mask=gt.copy())
        self.assertAlmostEqual(score.player_dice, 1.0)
        self.assertAlmostEqual(score.player_iou, 1.0)
        self.assertEqual(score.points_earned, 100)

    def test_empty_player(self):
        gt = np.zeros((16, 16), dtype=np.uint8)
        gt[2:8, 2:8] = 1
        player = np.zeros_like(gt)
        ai = gt.copy()
        score = score_training_round(ground_truth=gt, player_mask=player, ai_mask=ai)
        self.assertEqual(score.player_dice, 0.0)
        self.assertGreater(score.ai_dice, 0.9)


class StoreExportTests(unittest.TestCase):
    def test_submit_review_export_and_freeze_constants(self):
        self.assertEqual(E2B_THRESHOLD, 0.55)
        self.assertTrue(len(E2B_CHECKPOINT_SHA256) == 64)

        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(root=Path(tmp))
            rgb = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
            mask = np.zeros((48, 64), dtype=np.uint8)
            mask[10:30, 10:40] = 1
            ai = mask.copy()
            probs = np.linspace(0, 1, 48 * 64, dtype=np.float32).reshape(48, 64)
            img_bytes = b"fake-discovery-bytes-unique"

            c1 = store.submit(
                challenge_type="discovery",
                player_session_id="sess1",
                consent_to_contribute=True,
                player_mask=mask,
                image_rgb=rgb,
                image_bytes=img_bytes,
                ai_mask=ai,
                ai_probability=probs,
                image_source="unit-test",
                license_info="CC0 test",
                retain_image=True,
            )
            self.assertEqual(c1.verification_status, ContributionStatus.PENDING_REVIEW.value)
            self.assertEqual(c1.model_threshold, 0.55)

            # Duplicate
            c2 = store.submit(
                challenge_type="discovery",
                player_session_id="sess2",
                consent_to_contribute=True,
                player_mask=mask,
                image_rgb=rgb,
                image_bytes=img_bytes,
                ai_mask=ai,
                ai_probability=None,
                image_source="unit-test",
                license_info="CC0 test",
                retain_image=True,
            )
            self.assertEqual(c2.verification_status, ContributionStatus.DUPLICATE.value)

            store.review(c1.contribution_id, status="approved", comment="ok")
            # Discovery without weak gates: source/license already set → eligible
            manifest = export_approved_dataset(store, export_root=Path(tmp) / "exports")
            self.assertEqual(manifest["n_eligible_exported"], 1)
            self.assertIn("locked_test_split", manifest["forbidden"])
            self.assertIn("overwrite_existing_version", manifest["forbidden"])

    def test_oversized_upload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AnnotationStore(root=Path(tmp))
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:5, 2:5] = 1
            huge = b"x" * (9 * 1024 * 1024)
            with self.assertRaises(ValueError):
                store.submit(
                    challenge_type="discovery",
                    player_session_id="s",
                    consent_to_contribute=True,
                    player_mask=mask,
                    image_rgb=None,
                    image_bytes=huge,
                    ai_mask=None,
                    ai_probability=None,
                    image_source="x",
                    license_info="y",
                    retain_image=False,
                )


class GamificationTests(unittest.TestCase):
    def test_points_and_achievements(self):
        p = default_profile("abc")
        p = apply_training_score(
            p, challenge_id="train::x", points_earned=90, player_dice=0.9, beat_ai=True
        )
        self.assertEqual(p.points, 90)
        self.assertIn("first_hunt", p.achievements)
        self.assertIn("sharp_eye", p.achievements)
        self.assertIn("beat_ai", p.achievements)


class GroundTruthLoadTests(unittest.TestCase):
    def test_load_train_mask_if_present(self):
        challenges = load_training_challenges()
        if not challenges:
            self.skipTest("No local train/val frames available")
        ch = challenges[0]
        rgb = ch.load_rgb()
        gt = load_ground_truth_mask(ch.mask_path, target_hw=rgb.shape[:2])
        self.assertEqual(gt.shape, rgb.shape[:2])
        self.assertTrue(set(np.unique(gt)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
