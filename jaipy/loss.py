"""
Loss function for model training.
"""

import tensorflow as tf
from tensorflow.keras.losses import Loss  # pyright: reportMissingImports=false

# from jaipy import logger


def xywh_to_boxes(xywh: tf.Tensor) -> tf.Tensor:
    """
    Convert bounding box coordinates from (x, y, w, h) to (x1, y1, x2, y2).
    """
    [x, y, w, h] = [xywh[:, :, i, :] for i in range(4)]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return tf.stack([x1, y1, x2, y2], axis=2)


class YOLOLikeLoss(Loss):
    lambda_coord = 5
    lambda_noobj = 0.5

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        xywh_true = y_true[:, :, 1:, :]
        xywh_pred = y_pred[:, :, 1:, :]

        boxes_pred = xywh_to_boxes(xywh_pred)
        boxes_true = xywh_to_boxes(xywh_true)

        obj_true = y_true[:, :, 0, :]
        obj_pred = y_pred[:, :, 0, :]

        # TEMPOARY LOSS FUNCTION
        dummy_obj_loss = tf.reduce_sum(tf.square(obj_true - obj_pred))
        dummy_box_loss = tf.reduce_sum(tf.square(boxes_true - boxes_pred))

        return dummy_obj_loss + dummy_box_loss
