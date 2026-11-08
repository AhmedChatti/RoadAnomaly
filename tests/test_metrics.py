"""Unit tests for binary segmentation metrics and zero-denominator conventions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import binary_segmentation_metrics, threshold_probabilities  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_perfect_overlap(self):
        true = np.array([[0, 1], [1, 0]])
        pred = np.array([[0, 1], [1, 0]])
        m = binary_segmentation_metrics(true, pred)
        self.assertEqual(m["iou"], 1.0)
        self.assertEqual(m["dice"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["accuracy"], 1.0)

    def test_all_background_baseline(self):
        true = np.array([[0, 1], [1, 0]])
        pred = np.zeros_like(true)
        m = binary_segmentation_metrics(true, pred)
        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fn"], 2)
        self.assertEqual(m["iou"], 0.0)
        self.assertEqual(m["dice"], 0.0)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["accuracy"], 0.5)

    def test_both_empty_masks(self):
        true = np.zeros((4, 4), dtype=np.uint8)
        pred = np.zeros((4, 4), dtype=np.uint8)
        m = binary_segmentation_metrics(true, pred)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["iou"], 1.0)
        self.assertEqual(m["dice"], 1.0)

    def test_pred_empty_gt_positive(self):
        true = np.array([[1, 0], [0, 0]])
        pred = np.zeros_like(true)
        m = binary_segmentation_metrics(true, pred)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)

    def test_threshold(self):
        probs = np.array([[0.2, 0.8], [0.49, 0.5]])
        pred = threshold_probabilities(probs, 0.5)
        np.testing.assert_array_equal(pred, np.array([[0, 1], [0, 1]]))


if __name__ == "__main__":
    unittest.main()
