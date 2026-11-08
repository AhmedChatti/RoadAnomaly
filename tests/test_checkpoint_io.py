"""Tests for Keras checkpoint compatibility helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from checkpoint_io import (  # noqa: E402
    _strip_initializer_axes,
    load_keras_model_compat,
    load_weights_from_keras_zip,
)
from models.cnn import build_scratch_fcn  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402


class CheckpointIoTests(unittest.TestCase):
    def test_strip_initializer_axes(self):
        cfg = {
            "class_name": "GlorotUniform",
            "config": {"seed": None, "input_axes": None, "output_axes": None},
        }
        wrapper = {"layers": [{"kernel_initializer": cfg}]}
        _strip_initializer_axes(wrapper)
        self.assertNotIn("input_axes", cfg["config"])
        self.assertNotIn("output_axes", cfg["config"])
        self.assertIn("seed", cfg["config"])

    def test_roundtrip_scratch_fcn_via_compat_loader(self):
        model = build_scratch_fcn(input_height=32, input_width=32, base_filters=4, dropout=0.0)
        x = tf.zeros((1, 32, 32, 3))
        y0 = model.predict(x, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.keras"
            model.save(path)
            # Inject axes keys into archived config to simulate Keras 3.15 save.
            with zipfile.ZipFile(path, "r") as zin:
                names = zin.namelist()
                config_name = next(n for n in names if n.replace("\\", "/").endswith("config.json"))
                cfg = json.loads(zin.read(config_name).decode("utf-8"))
                _inject = {"seed": None, "input_axes": None, "output_axes": None}

                def inject(node):
                    if isinstance(node, dict):
                        if node.get("class_name") == "GlorotUniform" and isinstance(
                            node.get("config"), dict
                        ):
                            node["config"].update(_inject)
                        for v in node.values():
                            inject(v)
                    elif isinstance(node, list):
                        for item in node:
                            inject(item)

                inject(cfg)
                out = Path(tmp) / "injected.keras"
                with zipfile.ZipFile(out, "w") as zout:
                    for info in zin.infolist():
                        data = zin.read(info.filename)
                        if info.filename == config_name:
                            data = json.dumps(cfg).encode("utf-8")
                        zout.writestr(info, data)

            loaded = load_keras_model_compat(out, compile=False)
            y1 = loaded.predict(x, verbose=0)
            self.assertTrue(((y0 - y1) ** 2).mean() < 1e-10)

            rebuilt = build_scratch_fcn(
                input_height=32, input_width=32, base_filters=4, dropout=0.0
            )
            load_weights_from_keras_zip(rebuilt, path)
            y2 = rebuilt.predict(x, verbose=0)
            self.assertTrue(((y0 - y2) ** 2).mean() < 1e-10)


if __name__ == "__main__":
    unittest.main()
