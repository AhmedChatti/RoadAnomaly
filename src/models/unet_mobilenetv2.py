"""Phase 4 — U-Net with MobileNetV2 encoder (E2a / E2b).

Default construction uses weights=None so unit tests never download ImageNet files.
Training with ImageNet weights requires an explicit authorization flag at the script level.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

WeightsArg = Literal["imagenet"] | None

# Spatial hierarchy for 288×512 input (divisible by 32):
# block_1_expand_relu  → ~144×256 (1/2)
# block_3_expand_relu  → ~72×128  (1/4)
# block_6_expand_relu  → ~36×64   (1/8)
# block_13_expand_relu → ~18×32   (1/16)
# out_relu             → ~9×16    (1/32)
DEFAULT_SKIP_LAYERS = (
    "block_1_expand_relu",
    "block_3_expand_relu",
    "block_6_expand_relu",
    "block_13_expand_relu",
)
DEFAULT_BOTTLENECK = "out_relu"
DEFAULT_FINETUNE_FROM = "block_13_expand"

ENCODER_NAMES_ATTR = "_roadanomaly_encoder_layer_names"


def _decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Upsample + concat skip + dual 3×3 conv (no BatchNorm)."""
    x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    return x


def build_unet_mobilenetv2(
    *,
    input_height: int = 288,
    input_width: int = 512,
    weights: WeightsArg = None,
    encoder_trainable: bool = False,
    alpha: float = 1.0,
    decoder_filters: tuple[int, ...] = (256, 128, 64, 32),
    final_conv_filters: int = 16,
    dropout: float = 0.2,
    skip_layer_names: tuple[str, ...] = DEFAULT_SKIP_LAYERS,
    bottleneck_layer_name: str = DEFAULT_BOTTLENECK,
    name: str = "unet_mobilenetv2",
) -> keras.Model:
    """Build a U-Net with a MobileNetV2 encoder.

    Parameters
    ----------
    weights:
        None (tests / architecture) or "imagenet" (authorized training only).
    encoder_trainable:
        If False (E2a), all encoder layers are frozen after construction.
        E2b callers should pass True then call `configure_e2b_finetuning`.
    """
    if input_height % 32 != 0 or input_width % 32 != 0:
        raise ValueError(
            f"MobileNetV2 U-Net expects H,W divisible by 32; got {(input_height, input_width)}"
        )
    if weights not in (None, "imagenet"):
        raise ValueError(f"Unsupported weights={weights!r}; use None or 'imagenet'")

    inputs = keras.Input(shape=(input_height, input_width, 3), name="image")
    # Inputs must already use normalize="mobilenet_v2" (≈[-1, 1]).
    # Do NOT apply keras.applications.mobilenet_v2.preprocess_input again.
    base = MobileNetV2(
        include_top=False,
        weights=weights,
        input_tensor=inputs,
        alpha=alpha,
    )
    encoder_layer_names = {layer.name for layer in base.layers}

    skips = [base.get_layer(layer_name).output for layer_name in skip_layer_names]
    x = base.get_layer(bottleneck_layer_name).output

    # Decoder: 1/32 → 1/16 → 1/8 → 1/4 → 1/2 using skips in reverse order.
    skip_rev = list(reversed(skips))
    if len(decoder_filters) != len(skip_rev):
        raise ValueError(
            f"decoder_filters length {len(decoder_filters)} must match "
            f"skip count {len(skip_rev)}"
        )
    for index, (skip, filters) in enumerate(zip(skip_rev, decoder_filters)):
        x = _decoder_block(x, skip, filters, name=f"dec{index + 1}")

    # Final upsample 1/2 → full resolution.
    x = layers.UpSampling2D(2, interpolation="bilinear", name="dec_final_up")(x)
    x = layers.Conv2D(
        int(final_conv_filters), 3, padding="same", activation="relu", name="dec_final_conv1"
    )(x)
    x = layers.Conv2D(
        int(final_conv_filters), 3, padding="same", activation="relu", name="dec_final_conv2"
    )(x)
    x = layers.Dropout(dropout, name="dec_dropout")(x)

    prior = 0.1
    bias_init = -math.log((1.0 - prior) / prior)
    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        bias_initializer=keras.initializers.Constant(bias_init),
        name="anomaly_prob",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    setattr(model, ENCODER_NAMES_ATTR, encoder_layer_names)

    if not encoder_trainable:
        freeze_encoder(model, trainable=False)
        freeze_batch_norm(model)

    return model


def encoder_layer_names(model: keras.Model) -> set[str]:
    names = getattr(model, ENCODER_NAMES_ATTR, None)
    if names is None:
        raise RuntimeError(
            "Model is missing encoder layer metadata. "
            "Build with build_unet_mobilenetv2()."
        )
    return set(names)


def iter_encoder_layers(model: keras.Model) -> list[keras.layers.Layer]:
    names = encoder_layer_names(model)
    return [layer for layer in model.layers if layer.name in names]


def freeze_encoder(model: keras.Model, trainable: bool = False) -> None:
    for layer in iter_encoder_layers(model):
        layer.trainable = trainable


def freeze_batch_norm(model: keras.Model) -> int:
    """Force all BatchNormalization layers non-trainable. Returns count frozen."""
    count = 0
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            count += 1
    return count


def configure_e2b_finetuning(
    model: keras.Model,
    *,
    finetune_from_layer: str = DEFAULT_FINETUNE_FROM,
) -> dict[str, Any]:
    """Selectively unfreeze encoder layers from `finetune_from_layer` onward.

    Earlier encoder layers stay frozen. All BatchNormalization layers stay frozen.
    Decoder layers remain trainable.
    """
    encoder = iter_encoder_layers(model)
    if not encoder:
        raise RuntimeError("No MobileNetV2 encoder layers found on model")

    # Prefer exact layer-name match only (no silent prefix fallback).
    start_index = None
    for index, layer in enumerate(encoder):
        if layer.name == finetune_from_layer:
            start_index = index
            break
    if start_index is None:
        available = [layer.name for layer in encoder if "block_13" in layer.name]
        raise ValueError(
            f"Exact finetune_from_layer={finetune_from_layer!r} not found among encoder layers. "
            f"Nearby names: {available or [layer.name for layer in encoder[:20]]}"
        )

    for index, layer in enumerate(encoder):
        layer.trainable = index >= start_index

    bn_frozen = freeze_batch_norm(model)
    # Recompute after BN freeze so metadata matches actual trainable flags.
    frozen_names = [layer.name for layer in encoder if not layer.trainable]
    unfrozen_names = [layer.name for layer in encoder if layer.trainable]
    return {
        "finetune_from_layer": finetune_from_layer,
        "encoder_layers_total": len(encoder),
        "n_encoder_frozen": len(frozen_names),
        "n_encoder_unfrozen": len(unfrozen_names),
        "encoder_unfrozen": unfrozen_names,
        "batch_norm_frozen_count": bn_frozen,
    }


def encoder_trainable_flag_summary(model: keras.Model) -> dict[str, Any]:
    encoder = iter_encoder_layers(model)
    trainable = [layer.name for layer in encoder if layer.trainable]
    frozen = [layer.name for layer in encoder if not layer.trainable]
    bn_trainable = [
        layer.name
        for layer in model.layers
        if isinstance(layer, layers.BatchNormalization) and layer.trainable
    ]
    return {
        "n_encoder": len(encoder),
        "n_encoder_trainable": len(trainable),
        "n_encoder_frozen": len(frozen),
        "trainable_encoder_names": trainable,
        "batch_norm_still_trainable": bn_trainable,
    }
