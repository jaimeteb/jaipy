"""
Loss function for model training.
"""

# import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss  # pyright: reportMissingImports=false

# from jaipy.settings import settings
# from jaipy.logger import logger

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


class YOLOLikeLoss(Loss):
    lambda_coord = 5
    lambda_noobj = 0.5

    def call(
        self,
        Y_true_batch: tf.Tensor,
        Y_pred_batch: tf.Tensor,
    ) -> float:
        if Y_true_batch.shape[0] is None:
            return 0
        loss = 0
        for i in range(Y_true_batch.shape[0]):
            Y_true = Y_true_batch[i]
            Y_pred = Y_pred_batch[i]

            obj_true = Y_true[:, :, 0, :]
            x_true = Y_true[:, :, 1, :]
            y_true = Y_true[:, :, 2, :]
            w_true = Y_true[:, :, 3, :]
            h_true = Y_true[:, :, 4, :]

            obj_pred = Y_pred[:, :, 0, :]
            x_pred = Y_pred[:, :, 1, :]
            y_pred = Y_pred[:, :, 2, :]
            w_pred = Y_pred[:, :, 3, :]
            h_pred = Y_pred[:, :, 4, :]

            obj_both_idx = tf.where(tf.logical_and(obj_true > 0, obj_pred > 0))
            noobj_both_idx = tf.where(
                tf.logical_not(tf.logical_and(obj_true > 0, obj_pred > 0))
            )

            xy_diff = tf.square(x_true - x_pred) + tf.square(y_true - y_pred)
            wh_diff = tf.square(tf.sqrt(w_true) - tf.sqrt(w_pred)) + tf.square(
                tf.sqrt(h_true) - tf.sqrt(h_pred)
            )

            xy_loss = self.lambda_coord * tf.reduce_sum(
                tf.gather_nd(xy_diff, obj_both_idx)
            )
            wh_loss = self.lambda_coord * tf.reduce_sum(
                tf.gather_nd(wh_diff, obj_both_idx)
            )

            obj_diff = tf.square(obj_true - obj_pred)
            obj_loss = tf.reduce_sum(tf.gather_nd(obj_diff, obj_both_idx))
            noobj_loss = self.lambda_noobj * tf.reduce_sum(
                tf.gather_nd(obj_diff, noobj_both_idx)
            )

            # Add class loss
            loss += xy_loss + wh_loss + obj_loss + noobj_loss

        return loss
