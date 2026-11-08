"""Segmentation losses for imbalanced binary masks."""

from __future__ import annotations

import tensorflow as tf


def soft_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice)


def weighted_bce_loss(positive_weight: float = 9.0):
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Per-pixel weights: anomaly pixels get positive_weight.
        weights = 1.0 + (positive_weight - 1.0) * y_true
        # Manual weighted BCE for stability with Keras 3.
        eps = keras_epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce_map = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(weights * bce_map)

    loss.__name__ = "weighted_bce"
    return loss


def keras_epsilon() -> float:
    return float(tf.keras.backend.epsilon())


def bce_dice_loss(positive_weight: float = 9.0, dice_weight: float = 1.0):
    bce_fn = weighted_bce_loss(positive_weight=positive_weight)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return bce_fn(y_true, y_pred) + dice_weight * soft_dice_loss(y_true, y_pred)

    loss.__name__ = "bce_dice"
    return loss
