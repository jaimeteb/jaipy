"""
Loss function for model training.
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss  # pyright: reportMissingImports=false

from jaipy import utils


class YOLOLikeLoss(Loss):
    lambda_coord = 5
    lambda_noobj = 0.5

    def call(
        self,
        Y_true_batch: tf.Tensor,
        Y_pred_batch: tf.Tensor,
    ) -> float:
        with utils.device:
            obj_true, x_true, y_true, w_true, h_true = utils.get_obj_x_y_w_h(
                Y_true_batch
            )
            obj_pred, x_pred, y_pred, w_pred, h_pred = utils.get_obj_x_y_w_h(
                Y_pred_batch
            )

            obj_in_cell = tf.where(obj_true == 1)
            no_obj_in_cell = tf.where(obj_true == 0)

            xy_diff = K.square(x_true - x_pred) + K.square(y_true - y_pred)

            wh_diff = K.square(K.sqrt(w_true) - K.sqrt(w_pred)) + K.square(
                K.sqrt(h_true) - K.sqrt(h_pred)
            )

            obj_diff = K.square(obj_true - obj_pred)

            # class_pred_true = tf.nn.softmax(obj_true, axis=-1)
            # class_pred_pred = tf.nn.softmax(obj_pred, axis=-1)

            # class_diff = K.square(class_pred_true - class_pred_pred)

            return (
                self.lambda_coord * tf.reduce_sum(tf.gather_nd(xy_diff, obj_in_cell))
                + self.lambda_coord * tf.reduce_sum(tf.gather_nd(wh_diff, obj_in_cell))
                + tf.reduce_sum(tf.gather_nd(obj_diff, obj_in_cell))
                + self.lambda_noobj
                * tf.reduce_sum(tf.gather_nd(obj_diff, no_obj_in_cell))
                # + tf.reduce_sum(tf.gather_nd(class_diff, obj_in_cell))
            )
