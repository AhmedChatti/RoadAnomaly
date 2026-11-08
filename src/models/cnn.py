"""Experiment E1 — small fully convolutional segmenter trained from scratch."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    # No BatchNorm: with batch_size=2 on CPU, BN train/eval mismatch collapses
    # predictions toward background. Bias + ReLU is more stable here.
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    return x


def build_scratch_fcn(
    input_height: int = 288,
    input_width: int = 512,
    base_filters: int = 8,
    dropout: float = 0.2,
) -> keras.Model:
    """Encoder–decoder FCN without pretrained weights.

    Four downsampling stages (pool 2x2), then symmetric upsampling with
    skip concatenations (U-Net-lite). Output: sigmoid probability map.
    """
    inputs = keras.Input(shape=(input_height, input_width, 3), name="image")

    f1 = base_filters
    e1 = conv_block(inputs, f1, "enc1")
    p1 = layers.MaxPooling2D(2, name="pool1")(e1)

    e2 = conv_block(p1, f1 * 2, "enc2")
    p2 = layers.MaxPooling2D(2, name="pool2")(e2)

    e3 = conv_block(p2, f1 * 4, "enc3")
    p3 = layers.MaxPooling2D(2, name="pool3")(e3)

    e4 = conv_block(p3, f1 * 8, "enc4")
    p4 = layers.MaxPooling2D(2, name="pool4")(e4)

    bottleneck = conv_block(p4, f1 * 16, "bottleneck")
    bottleneck = layers.Dropout(dropout, name="bottleneck_dropout")(bottleneck)

    u4 = layers.UpSampling2D(2, interpolation="bilinear", name="up4")(bottleneck)
    u4 = layers.Concatenate(name="skip4")([u4, e4])
    d4 = conv_block(u4, f1 * 8, "dec4")

    u3 = layers.UpSampling2D(2, interpolation="bilinear", name="up3")(d4)
    u3 = layers.Concatenate(name="skip3")([u3, e3])
    d3 = conv_block(u3, f1 * 4, "dec3")

    u2 = layers.UpSampling2D(2, interpolation="bilinear", name="up2")(d3)
    u2 = layers.Concatenate(name="skip2")([u2, e2])
    d2 = conv_block(u2, f1 * 2, "dec2")

    u1 = layers.UpSampling2D(2, interpolation="bilinear", name="up1")(d2)
    u1 = layers.Concatenate(name="skip1")([u1, e1])
    d1 = conv_block(u1, f1, "dec1")
    d1 = layers.Dropout(dropout, name="dec_dropout")(d1)

    # Bias ≈ logit(prior) with prior≈0.1 so early predictions are not all-negative.
    import math

    prior = 0.1
    bias_init = -math.log((1.0 - prior) / prior)
    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        bias_initializer=keras.initializers.Constant(bias_init),
        name="anomaly_prob",
    )(d1)
    model = keras.Model(inputs=inputs, outputs=outputs, name="scratch_fcn_unetlite")
    return model
