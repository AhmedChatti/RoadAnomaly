"""Unit tests for deterministic preprocessing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import (  # noqa: E402
    assert_image_mask_alignment,
    normalize_image,
    preprocess_pair,
    remap_official_mask_to_binary,
    resize_image_rgb,
    resize_mask_nearest,
)


class PreprocessingTests(unittest.TestCase):
    def test_remap_official_values(self):
        mask = np.array([[0, 2], [2, 0]], dtype=np.uint8)
        binary = remap_official_mask_to_binary(mask)
        np.testing.assert_array_equal(binary, np.array([[0, 1], [1, 0]]))

    def test_remap_prepared_255(self):
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        binary = remap_official_mask_to_binary(mask)
        np.testing.assert_array_equal(binary, np.array([[0, 1], [1, 0]]))

    def test_remap_rejects_unknown_values(self):
        mask = np.array([[0, 3]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            remap_official_mask_to_binary(mask)

    def test_alignment_check(self):
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        mask = np.zeros((4, 5), dtype=np.uint8)
        with self.assertRaises(ValueError):
            assert_image_mask_alignment(image, mask, frame_id="demo")

    def test_mask_nearest_keeps_binary_labels(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 2
        resized = resize_mask_nearest(remap_official_mask_to_binary(mask), 5, 5)
        self.assertEqual(set(np.unique(resized).tolist()), {0, 1})

    def test_image_resize_shape(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        resized = resize_image_rgb(image, 512, 288)
        self.assertEqual(resized.shape, (288, 512, 3))

    def test_preprocess_pair_output(self):
        image = np.random.default_rng(0).integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
        mask = np.zeros((720, 1280), dtype=np.uint8)
        mask[100:200, 100:300] = 2
        packed = preprocess_pair(image, mask, normalize="scale_0_1", frame_id="synthetic")
        self.assertEqual(packed["image"].shape, (288, 512, 3))
        self.assertEqual(packed["mask"].shape, (288, 512))
        self.assertEqual(set(np.unique(packed["mask"]).tolist()), {0, 1})
        self.assertGreater(packed["anomaly_pixels"], 0)
        self.assertTrue(np.all(packed["image"] >= 0.0) and np.all(packed["image"] <= 1.0))

    def test_imagenet_normalize_finite(self):
        image = np.full((8, 8, 3), 128, dtype=np.uint8)
        out = normalize_image(image, "imagenet")
        self.assertTrue(np.isfinite(out).all())

    def test_empty_mask_rejected(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        with self.assertRaises(ValueError):
            preprocess_pair(image, mask)


class RealFrameSmokeTest(unittest.TestCase):
    def test_one_real_frame(self):
        frame_id = "animals17_E6_near_Kvaenangen"
        image_path = ROOT / "frames" / f"{frame_id}.webp"
        mask_path = ROOT / "frames" / f"{frame_id}.labels" / "labels_semantic.png"
        if not image_path.exists():
            self.skipTest("dataset frames not present")
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path))
        packed = preprocess_pair(image, mask, normalize="scale_0_1", frame_id=frame_id)
        self.assertEqual(packed["image"].shape, (288, 512, 3))
        self.assertEqual(set(np.unique(packed["mask"]).tolist()), {0, 1})
        self.assertGreater(packed["anomaly_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
