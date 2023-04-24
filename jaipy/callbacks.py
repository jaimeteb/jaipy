"""
Custom raining callbacks
"""
import io

import tensorflow as tf

from jaipy import utils
from jaipy.dataset import DataGenerator


class ImagePredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_name: str, test_data: DataGenerator, test_batch_size: int):
        super().__init__()
        self.model_name = model_name
        self.test_batch_size = test_batch_size

        self.X, self.Y_true = test_data[0]

    def on_epoch_end(self, epoch, logs={}):  # pylint: disable=dangerous-default-value
        logdir = f"./logs/{self.model_name}/images"
        file_writer = tf.summary.create_file_writer(logdir)

        X, Y_true = self.X, self.Y_true
        Y_pred = self.model.predict(X, verbose=1)

        with file_writer.as_default():
            for idx in range(self.test_batch_size):
                img = utils.draw_prediction_and_truth(X[idx], Y_pred[idx], Y_true[idx])

                img_bytes = io.BytesIO()
                img.save(img_bytes, format="png")
                tf_image = tf.image.decode_png(img_bytes.getvalue(), channels=3)
                tf_image = tf.expand_dims(tf_image, 0)
                img_bytes.close()

                tf.summary.image(f"sample-{idx}", tf_image, step=epoch)
