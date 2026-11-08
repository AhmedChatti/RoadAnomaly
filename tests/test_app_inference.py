"""Unit tests for Streamlit inference helpers (no test-set access)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app_inference import (  # noqa: E402
    E2B_INPUT_HEIGHT,
    E2B_INPUT_WIDTH,
    E2B_THRESHOLD,
    EXPLORATION_LABEL,
    colorize_mask,
    derive_exploratory_view,
    encode_png_mask,
    encode_png_probability,
    encode_png_rgb,
    load_user_image,
    make_overlay,
    predict_e2b,
    preprocess_for_e2b,
    probability_heatmap_rgb,
    resize_probability_bilinear,
)
from metrics import threshold_probabilities  # noqa: E402


class FakeModel:
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, E2B_INPUT_HEIGHT, E2B_INPUT_WIDTH, 3)
        # Synthetic probability map in (0,1)
        probs = np.linspace(0.0, 1.0, E2B_INPUT_HEIGHT * E2B_INPUT_WIDTH, dtype=np.float32)
        probs = probs.reshape(E2B_INPUT_HEIGHT, E2B_INPUT_WIDTH)
        return probs[None, ..., None]


class AppInferenceTests(unittest.TestCase):
    def test_preprocess_dimensions_and_mobilenet_range(self):
        image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        batch, model_rgb = preprocess_for_e2b(image)
        self.assertEqual(model_rgb.shape, (E2B_INPUT_HEIGHT, E2B_INPUT_WIDTH, 3))
        self.assertEqual(batch.shape, (1, E2B_INPUT_HEIGHT, E2B_INPUT_WIDTH, 3))
        self.assertTrue(batch.min() >= -1.0 - 1e-5)
        self.assertTrue(batch.max() <= 1.0 + 1e-5)

    def test_threshold_freeze_value(self):
        probs = np.array([[0.54, 0.55, 0.56]], dtype=np.float32)
        binary = threshold_probabilities(probs, E2B_THRESHOLD)
        np.testing.assert_array_equal(binary, np.array([[0, 1, 1]], dtype=np.uint8))

    def test_overlay_and_encoders(self):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        image[:] = (10, 20, 30)
        mask = np.zeros((32, 48), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        overlay = make_overlay(image, mask)
        self.assertEqual(overlay.shape, image.shape)
        self.assertEqual(colorize_mask(mask).shape, (32, 48, 3))
        heat = probability_heatmap_rgb(np.linspace(0, 1, 32 * 48, dtype=np.float32).reshape(32, 48))
        self.assertEqual(heat.shape, (32, 48, 3))
        self.assertTrue(encode_png_rgb(overlay).startswith(b"\x89PNG"))
        self.assertTrue(encode_png_mask(mask).startswith(b"\x89PNG"))
        probs = np.linspace(0, 1, 32 * 48, dtype=np.float32).reshape(32, 48)
        self.assertTrue(encode_png_probability(probs).startswith(b"\x89PNG"))

    def test_invalid_upload_extension(self):
        with self.assertRaises(ValueError):
            load_user_image(b"not-an-image", filename="bad.gif")

    def test_predict_pipeline_with_fake_model(self):
        image = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        result = predict_e2b(FakeModel(), image)
        self.assertEqual(result.binary_mask.shape, (E2B_INPUT_HEIGHT, E2B_INPUT_WIDTH))
        self.assertEqual(result.binary_full.shape, (360, 640))
        self.assertEqual(result.probability_full.shape, (360, 640))
        self.assertEqual(result.overlay_rgb.shape, (360, 640, 3))
        self.assertEqual(result.threshold, E2B_THRESHOLD)
        self.assertGreaterEqual(result.anomaly_pixel_pct, 0.0)
        self.assertLessEqual(result.anomaly_pixel_pct, 100.0)
        self.assertGreaterEqual(result.mean_anomaly_probability, 0.0)
        self.assertLessEqual(result.mean_anomaly_probability, 1.0)

    def test_probability_resize_shape(self):
        probs = np.zeros((E2B_INPUT_HEIGHT, E2B_INPUT_WIDTH), dtype=np.float32)
        out = resize_probability_bilinear(probs, 100, 50)
        self.assertEqual(out.shape, (50, 100))

    def test_exploration_threshold_cannot_change_official_prediction(self):
        image = np.random.randint(0, 255, (200, 320, 3), dtype=np.uint8)
        official = predict_e2b(FakeModel(), image)
        snap_mask = official.binary_full.copy()
        snap_model_mask = official.binary_mask.copy()
        snap_overlay = official.overlay_rgb.copy()
        snap_probs = official.probability_map.copy()
        snap_pct = official.anomaly_pixel_pct
        snap_thr = official.threshold
        snap_mean = official.mean_anomaly_probability

        for thr in (0.10, 0.25, 0.40, 0.70, 0.90):
            explore = derive_exploratory_view(official, thr)
            self.assertEqual(explore.threshold, thr)
            self.assertEqual(explore.label, EXPLORATION_LABEL)
            # Official frozen fields remain identical after exploration.
            self.assertEqual(official.threshold, E2B_THRESHOLD)
            self.assertEqual(official.threshold, snap_thr)
            self.assertEqual(official.anomaly_pixel_pct, snap_pct)
            self.assertEqual(official.mean_anomaly_probability, snap_mean)
            np.testing.assert_array_equal(official.binary_full, snap_mask)
            np.testing.assert_array_equal(official.binary_mask, snap_model_mask)
            np.testing.assert_array_equal(official.overlay_rgb, snap_overlay)
            np.testing.assert_array_equal(official.probability_map, snap_probs)
            # Exploration is a separate object.
            if abs(thr - E2B_THRESHOLD) > 1e-9:
                self.assertFalse(np.array_equal(explore.binary_full, official.binary_full))

    def test_predict_e2b_has_no_threshold_override(self):
        import inspect

        params = inspect.signature(predict_e2b).parameters
        self.assertNotIn("threshold", params)


class OptionalRealCheckpointTests(unittest.TestCase):
    def test_real_e2b_load_if_present(self):
        from app_inference import resolve_e2b_checkpoint, load_e2b_model, predict_e2b

        try:
            path = resolve_e2b_checkpoint()
        except FileNotFoundError:
            self.skipTest("E2b checkpoint not available locally")
        model = load_e2b_model(path, verify_sha=True)
        # Synthetic image only — never use held-out test frames for app tuning.
        image = np.zeros((288, 512, 3), dtype=np.uint8)
        image[:, :] = (80, 90, 100)
        result = predict_e2b(model, image)
        self.assertEqual(result.probability_map.shape, (E2B_INPUT_HEIGHT, E2B_INPUT_WIDTH))
        self.assertTrue(np.isfinite(result.probability_map).all())


if __name__ == "__main__":
    unittest.main()
