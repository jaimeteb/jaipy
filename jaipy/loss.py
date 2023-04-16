"""
Loss function for model training.
"""

import typing as t

# import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss  # pyright: reportMissingImports=false

# from jaipy.settings import settings

# def xywh_to_boxes(tensor: tf.Tensor) -> tf.Tensor:
#     """
#     Convert bounding box coordinates from (x, y, w, h) to (x1, y1, x2, y2).
#     """

#     arr = np.zeros_like(tensor)

#     for ci in range(tensor.shape[0]):
#         for cj in range(tensor.shape[1]):
#             for idx in range(tensor.shape[3]):
#                 if tensor[ci, cj, 0, idx] > 0:
#                     x = (tensor[ci, cj, 1, idx] + ci) / settings.grid
#                     y = (tensor[ci, cj, 2, idx] + cj) / settings.grid
#                     w = tensor[ci, cj, 3, idx]
#                     h = tensor[ci, cj, 4, idx]

#                     x1 = x - w / 2
#                     y1 = y - h / 2
#                     x2 = x + w / 2
#                     y2 = y + h / 2

#                     arr[ci, cj, 1, idx] = x1
#                     arr[ci, cj, 2, idx] = y1
#                     arr[ci, cj, 3, idx] = x2
#                     arr[ci, cj, 4, idx] = y2

#     return tf.convert_to_tensor(arr)[:, :, 1:, :]


def get_obj_x_y_w_h(
    tensor: tf.Tensor,
) -> t.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Get object, x, y, w, h from tensor.
    """

    obj = tensor[..., 0]
    x = tensor[..., 1]
    y = tensor[..., 2]
    w = tensor[..., 3]
    h = tensor[..., 4]

    return obj, x, y, w, h


class YOLOLikeLoss(Loss):
    lambda_coord = 5
    lambda_noobj = 0.5

    def call(
        self,
        Y_true_batch: tf.Tensor,
        Y_pred_batch: tf.Tensor,
    ) -> float:
        obj_true, x_true, y_true, w_true, h_true = get_obj_x_y_w_h(Y_true_batch)
        obj_pred, x_pred, y_pred, w_pred, h_pred = get_obj_x_y_w_h(Y_pred_batch)

        xy_diff = K.square(x_true - x_pred) + K.square(y_true - y_pred)

        wh_diff = K.square(K.sqrt(w_true) - K.sqrt(w_pred)) + K.square(
            K.sqrt(h_true) - K.sqrt(h_pred)
        )

        obj_diff = K.square(obj_true - obj_pred)

        loss = (
            self.lambda_coord * tf.reduce_sum(xy_diff)
            + self.lambda_coord * tf.reduce_sum(wh_diff)
            + tf.reduce_sum(obj_diff)
        )

        return loss
