"""Load `.keras` checkpoints across minor Keras serializer mismatches.

E1 was saved on Keras 3.15.1 (GlorotUniform config includes `input_axes` /
`output_axes`). Some Colab runtimes ship an older GlorotUniform that rejects
those keys. Strip them from config.json inside the archive and reload.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from tensorflow import keras

_AXES_KEYS = ("input_axes", "output_axes")
_INIT_CLASS_NAMES = {
    "GlorotUniform",
    "GlorotNormal",
    "HeNormal",
    "HeUniform",
    "LecunNormal",
    "LecunUniform",
    "VarianceScaling",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "Orthogonal",
}


def _strip_initializer_axes(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("class_name") in _INIT_CLASS_NAMES and isinstance(node.get("config"), dict):
            for key in _AXES_KEYS:
                node["config"].pop(key, None)
        for value in node.values():
            _strip_initializer_axes(value)
    elif isinstance(node, list):
        for item in node:
            _strip_initializer_axes(item)


def _rewrite_keras_zip_strip_axes(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            name = info.filename.replace("\\", "/")
            if name.endswith("config.json"):
                cfg = json.loads(data.decode("utf-8"))
                _strip_initializer_axes(cfg)
                data = json.dumps(cfg).encode("utf-8")
            zout.writestr(info, data)


def load_weights_from_keras_zip(model: keras.Model, path: str | Path) -> None:
    """Load weight tensors from a full `.keras` archive into an existing model."""
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(path, "r") as zin:
            names = {n.replace("\\", "/"): n for n in zin.namelist()}
            weight_key = next(
                (logical for logical in names if logical.endswith("model.weights.h5")),
                None,
            )
            if weight_key is None:
                raise FileNotFoundError(f"No model.weights.h5 inside {path}")
            extracted = tmp_path / "model.weights.h5"
            extracted.write_bytes(zin.read(names[weight_key]))
        model.load_weights(extracted)


def load_keras_model_compat(
    path: str | Path,
    *,
    custom_objects: dict[str, Any] | None = None,
    compile: bool = False,
) -> keras.Model:
    """`load_model` with a one-shot config rewrite for initializer axes mismatch."""
    path = Path(path)
    try:
        return keras.models.load_model(
            path, custom_objects=custom_objects, compile=compile
        )
    except TypeError as exc:
        text = str(exc)
        if "input_axes" not in text and "output_axes" not in text:
            raise
    with tempfile.TemporaryDirectory() as tmp:
        sanitized = Path(tmp) / "sanitized.keras"
        _rewrite_keras_zip_strip_axes(path, sanitized)
        return keras.models.load_model(
            sanitized, custom_objects=custom_objects, compile=compile
        )
