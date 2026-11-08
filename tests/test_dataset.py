"""Tests for split locking, dataset shapes and train-only augmentation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from augmentation import AugmentationConfig, augment_pair  # noqa: E402
from dataset import RoadAnomalyDataset, load_split_manifest, verify_split_lock  # noqa: E402
from preprocessing import load_rgb_image, load_semantic_mask, preprocess_pair  # noqa: E402


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "frames").exists():
            raise unittest.SkipTest("frames/ missing")
        cls.manifest = load_split_manifest()

    def test_split_counts_and_lock(self):
        counts = self.manifest["split"].value_counts().to_dict()
        self.assertEqual(counts["train"], 42)
        self.assertEqual(counts["val"], 9)
        self.assertEqual(counts["test"], 9)
        digest = verify_split_lock(self.manifest)
        self.assertEqual(len(digest), 64)

    def test_no_scene_group_leakage(self):
        crossing = int(self.manifest.groupby("scene_group")["split"].nunique().gt(1).sum())
        self.assertEqual(crossing, 0)

    def test_dataset_shapes_all_splits(self):
        for split in ("train", "val", "test"):
            data = RoadAnomalyDataset(split, normalize="scale_0_1", augment=False)
            self.assertGreater(len(data), 0)
            sample = data[0]
            self.assertEqual(sample["image"].shape, (288, 512, 3))
            self.assertEqual(sample["mask"].shape, (288, 512, 1))
            self.assertEqual(set(np.unique(sample["mask_uint8"]).tolist()), {0, 1})
            self.assertGreater(sample["anomaly_pixels"], 0)

    def test_augmentation_forbidden_on_val(self):
        data = RoadAnomalyDataset("val", augment=True)
        self.assertFalse(data.augment)
        sample = data[0]
        # even if caller asked augment=True, val must stay unaugmented path
        self.assertEqual(sample["split"], "val")

    def test_augmentation_keeps_binary_mask(self):
        data = RoadAnomalyDataset("train", augment=False)
        sample = data[0]
        rng = np.random.default_rng(123)
        cfg = AugmentationConfig.from_project_config()
        image_a, mask_a = augment_pair(sample["image_uint8"], sample["mask_uint8"], rng=rng, config=cfg)
        self.assertEqual(image_a.shape, sample["image_uint8"].shape)
        self.assertEqual(set(np.unique(mask_a).tolist()) - {0, 1}, set())
        self.assertGreater(int(mask_a.sum()), 0)

    def test_batch_iteration(self):
        data = RoadAnomalyDataset("val", augment=False)
        batches = list(data.iter_batches(batch_size=2, shuffle=False))
        x0, y0 = batches[0]
        self.assertEqual(x0.ndim, 4)
        self.assertEqual(y0.ndim, 4)
        self.assertEqual(x0.shape[1:], (288, 512, 3))
        self.assertEqual(y0.shape[1:], (288, 512, 1))


class ConsistencyOriginalVsPipeline(unittest.TestCase):
    def test_mask_still_overlaps_object_region(self):
        frame_id = "lost_tires08"
        image_path = ROOT / "frames" / f"{frame_id}.webp"
        mask_path = ROOT / "frames" / f"{frame_id}.labels" / "labels_semantic.png"
        if not image_path.exists():
            self.skipTest("dataset frames not present")
        image = load_rgb_image(image_path)
        mask = load_semantic_mask(mask_path)
        packed = preprocess_pair(image, mask, frame_id=frame_id)
        # Tiny anomaly must survive resize.
        self.assertGreaterEqual(packed["anomaly_pixels"], 50)
        self.assertLess(packed["anomaly_ratio"], 0.05)


if __name__ == "__main__":
    unittest.main()
