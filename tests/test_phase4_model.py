"""Phase 4 model tests — never download ImageNet weights."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from losses import bce_dice_loss  # noqa: E402
from models.unet_mobilenetv2 import (  # noqa: E402
    build_unet_mobilenetv2,
    configure_e2b_finetuning,
    encoder_trainable_flag_summary,
    freeze_batch_norm,
)
from phase4_training import build_phase4_model_for_experiment, load_phase4_config  # noqa: E402
from preprocessing import normalize_image  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402


class Phase4ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.height = 288
        cls.width = 512
        cls.model = build_unet_mobilenetv2(
            input_height=cls.height,
            input_width=cls.width,
            weights=None,
            encoder_trainable=False,
        )

    def test_output_shape(self):
        x = np.zeros((2, self.height, self.width, 3), dtype=np.float32)
        y = self.model.predict(x, verbose=0)
        self.assertEqual(y.shape, (2, self.height, self.width, 1))

    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 255, size=(self.height, self.width, 3), dtype=np.uint8)
        x = normalize_image(raw, "mobilenet_v2")[None, ...]
        y = self.model.predict(x, verbose=0)
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)

    def test_e2a_encoder_fully_frozen(self):
        summary = encoder_trainable_flag_summary(self.model)
        self.assertEqual(summary["n_encoder_trainable"], 0)
        self.assertGreater(summary["n_encoder_frozen"], 0)
        self.assertEqual(summary["batch_norm_still_trainable"], [])

    def test_e2b_selective_finetune_and_bn_frozen(self):
        from models.unet_mobilenetv2 import iter_encoder_layers

        model, info = build_phase4_model_for_experiment(
            "E2b", weights=None, authorize_imagenet_download=False
        )
        finetune = info["finetune"]
        self.assertIsNotNone(finetune)
        self.assertGreater(finetune["n_encoder_unfrozen"], 0)
        self.assertGreater(finetune["n_encoder_frozen"], 0)
        summary = encoder_trainable_flag_summary(model)
        self.assertEqual(summary["batch_norm_still_trainable"], [])

        # Inspect actual layer.trainable flags (not only helper metadata).
        encoder = iter_encoder_layers(model)
        names = [layer.name for layer in encoder]
        self.assertIn("block_13_expand", names)
        start = names.index("block_13_expand")
        for layer in encoder[:start]:
            self.assertFalse(layer.trainable, layer.name)
        for layer in encoder[start:]:
            if isinstance(layer, layers.BatchNormalization):
                self.assertFalse(layer.trainable, layer.name)
            else:
                self.assertTrue(layer.trainable, layer.name)
        for layer in model.layers:
            if isinstance(layer, layers.BatchNormalization):
                self.assertFalse(layer.trainable, layer.name)

        decoder_layers = [
            layer
            for layer in model.layers
            if layer.name.startswith("dec") or layer.name == "anomaly_prob"
        ]
        self.assertTrue(any(layer.trainable for layer in decoder_layers))
        self.assertGreater(len(model.trainable_weights), 0)

    def test_e2b_exact_layer_match_required(self):
        model = build_unet_mobilenetv2(weights=None, encoder_trainable=True)
        # Prefix-only names must not silently match.
        with self.assertRaises(ValueError) as ctx:
            configure_e2b_finetuning(model, finetune_from_layer="block_13")
        self.assertIn("Exact finetune_from_layer", str(ctx.exception))

    def test_no_unexpected_trainable_encoder_in_e2a_builder(self):
        model, _ = build_phase4_model_for_experiment(
            "E2a", weights=None, authorize_imagenet_download=False
        )
        summary = encoder_trainable_flag_summary(model)
        self.assertEqual(summary["trainable_encoder_names"], [])

    def test_loss_compatibility_and_decoder_gradients(self):
        model, _ = build_phase4_model_for_experiment(
            "E2a", weights=None, authorize_imagenet_download=False
        )
        loss_fn = bce_dice_loss(positive_weight=15.0)
        x = tf.zeros((1, self.height, self.width, 3), dtype=tf.float32)
        flat = tf.reshape(tf.range(self.height * self.width), (1, self.height, self.width, 1))
        y_true = tf.cast(flat % 17 == 0, tf.float32)
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y_true, y_pred)
        grads = tape.gradient(loss, model.trainable_weights)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertGreater(len(model.trainable_weights), 0)
        nonempty = [g for g in grads if g is not None]
        self.assertGreater(len(nonempty), 0)
        self.assertTrue(all(tf.reduce_all(tf.math.is_finite(g)).numpy() for g in nonempty))

    def test_serialization_roundtrip(self):
        # Prefer in-memory weight roundtrip to avoid filling a nearly-full disk
        # with a full MobileNetV2 .keras zip during CI/local unit tests.
        x = np.zeros((1, self.height, self.width, 3), dtype=np.float32)
        y1 = self.model.predict(x, verbose=0)
        weights = self.model.get_weights()
        clone = build_unet_mobilenetv2(
            input_height=self.height,
            input_width=self.width,
            weights=None,
            encoder_trainable=False,
        )
        clone.set_weights(weights)
        y2 = clone.predict(x, verbose=0)
        np.testing.assert_allclose(y1, y2, rtol=1e-5, atol=1e-5)

        # Optional on-disk save when enough free space is available (~150 MB).
        usage = shutil.disk_usage(tempfile.gettempdir())
        if usage.free < 150 * 1024 * 1024:
            self.skipTest(f"Insufficient disk free space for .keras save ({usage.free} bytes)")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase4_smoke.keras"
            self.model.save(path)
            reloaded = keras.models.load_model(path, compile=False)
            y3 = reloaded.predict(x, verbose=0)
            np.testing.assert_allclose(y1, y3, rtol=1e-5, atol=1e-5)

    def test_freeze_batch_norm_helper(self):
        count = freeze_batch_norm(self.model)
        self.assertGreater(count, 0)
        for layer in self.model.layers:
            if isinstance(layer, layers.BatchNormalization):
                self.assertFalse(layer.trainable)

    def test_configure_e2b_rejects_unknown_layer(self):
        model = build_unet_mobilenetv2(weights=None, encoder_trainable=True)
        with self.assertRaises(ValueError):
            configure_e2b_finetuning(model, finetune_from_layer="not_a_real_layer")

    def test_phase4_config_loads(self):
        p4 = load_phase4_config()
        self.assertEqual(p4["input"]["normalize"], "mobilenet_v2")
        self.assertEqual(p4["input"]["height"], 288)
        self.assertEqual(p4["input"]["width"], 512)
        self.assertFalse(p4["safety"]["allow_test_access"])
        self.assertEqual(p4["loss"]["positive_weight"], 15.0)
        self.assertEqual(p4["decoder"]["final_conv_filters"], 16)
        self.assertTrue(str(p4["artefacts"]["metadata_subdir"]).endswith("phase4"))

    def test_final_conv_filters_consumed_from_argument(self):
        model = build_unet_mobilenetv2(
            weights=None,
            encoder_trainable=False,
            final_conv_filters=8,
        )
        layer = model.get_layer("dec_final_conv1")
        self.assertEqual(int(layer.filters), 8)


class Phase4PreprocessCompatibilityTests(unittest.TestCase):
    def test_mobilenet_v2_normalize_range(self):
        image = np.array([[[0, 127, 255]]], dtype=np.uint8)
        out = normalize_image(image, "mobilenet_v2")
        self.assertAlmostEqual(float(out[0, 0, 0]), -1.0, places=5)
        self.assertAlmostEqual(float(out[0, 0, 2]), 1.0, places=5)
        self.assertTrue(-1.0 < float(out[0, 0, 1]) < 0.0)

    def test_imagenet_mode_unchanged_for_e1(self):
        image = np.full((2, 2, 3), 255, dtype=np.uint8)
        out = normalize_image(image, "imagenet")
        self.assertTrue(np.isfinite(out).all())
        self.assertFalse(np.allclose(out, image.astype(np.float32) / 127.5 - 1.0))


if __name__ == "__main__":
    unittest.main()
