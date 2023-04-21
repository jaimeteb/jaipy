"""
Convolutional Neural Network model definition
"""
# pyright: reportMissingImports=false

import datetime as dt
import typing as t

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    MaxPool2D,
    Reshape,
)
from tensorflow.keras.regularizers import l2

from jaipy import loss, utils
from jaipy.dataset import DataGenerator
from jaipy.logger import logger
from jaipy.settings import settings

tf.keras.backend.set_floatx("float64")


def Convolutional(  # pylint: disable=too-many-arguments
    layer: t.Any,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    batch_norm: bool = True,
    padding: str = "same",
):
    layer = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_regularizer=l2(0.000_5),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0),
    )(layer)
    layer = BatchNormalization()(layer) if batch_norm else layer
    layer = LeakyReLU(alpha=0.1)(layer)

    return layer


def MaxPooling(
    pool_size: int,
    strides: int,
    padding: str,
):
    return MaxPool2D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
    )


class Model:  # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        input_size: int = settings.input_size,
        channels: int = settings.channels,
        num_classes: int = settings.num_classes,
        num_boxes: int = settings.num_boxes,
        batch_size: int = settings.batch_size,
        epochs: int = settings.epochs,
        learning_rate: float = settings.learning_rate,
        grid: int = settings.grid,
    ):
        self.input_size = input_size
        self.channels = channels
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.grid = grid

        self.model = self._create_model()

    def _create_model(self) -> tf.keras.Model:
        input_layer = Input([self.input_size, self.input_size, self.channels])

        # Darknet-like architecture
        layer = Convolutional(input_layer, 64, 7, 2)
        layer = MaxPooling(2, 2, "same")(layer)

        layer = Convolutional(layer, 192, 3)
        layer = MaxPooling(2, 2, "same")(layer)

        layer = Convolutional(layer, 128, 1)
        layer = Convolutional(layer, 256, 3)
        layer = Convolutional(layer, 256, 1)
        layer = Convolutional(layer, 512, 3)
        layer = MaxPooling(2, 2, "same")(layer)

        layer = Convolutional(layer, 256, 1)  #
        layer = Convolutional(layer, 512, 3)
        # layer = Convolutional(layer, 256, 1)
        # layer = Convolutional(layer, 512, 3)
        # layer = Convolutional(layer, 256, 1)
        # layer = Convolutional(layer, 512, 3)
        # layer = Convolutional(layer, 256, 1)
        # layer = Convolutional(layer, 512, 3)  #
        layer = Convolutional(layer, 512, 1)
        layer = Convolutional(layer, 1024, 3)
        layer = MaxPooling(2, 2, "same")(layer)

        layer = Convolutional(layer, 512, 1)  #
        layer = Convolutional(layer, 1024, 3)
        # layer = Convolutional(layer, 512, 1)
        # layer = Convolutional(layer, 1024, 3)  #
        layer = Convolutional(layer, 1024, 3)
        layer = Convolutional(layer, 1024, 3, 2)

        layer = Convolutional(layer, 1024, 3)
        layer = Convolutional(layer, 1024, 3)

        # Fully connected layers
        layer = Flatten()(layer)
        layer = Dense(4096, activation=LeakyReLU(alpha=0.1))(layer)
        layer = Dropout(rate=0.5)(layer)
        layer = Dense(
            self.grid * self.grid * 5 * self.num_classes,
            activation=LeakyReLU(alpha=0.1),
        )(layer)
        final_layer = Reshape((self.grid, self.grid, 5, self.num_classes))(layer)

        # Create model
        model = tf.keras.Model(input_layer, final_layer)
        model.summary(print_fn=logger.info)

        return model

    def train(
        self,
        train_data: DataGenerator,
        val_data: DataGenerator,
        checkpoints: bool = True,
    ) -> None:
        with utils.device:
            yolo_like_loss = loss.YOLOLikeLoss()
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                # loss=tf.keras.losses.MeanSquaredError(),
                loss=yolo_like_loss,
            )

            model_name = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

            callbacks = [
                tf.keras.callbacks.TensorBoard(
                    log_dir=f"./logs/{model_name}",
                    histogram_freq=1,
                )
            ]

            if checkpoints:
                callbacks.append(
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=f"./models/{model_name}" + "-{epoch:02d}.h5",
                        save_best_only=False,
                        save_weights_only=True,
                    )
                )

            self.model.fit(
                train_data,
                validation_data=val_data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=1,
                callbacks=callbacks,
            )
            self.model.save(f"./models/{model_name}.h5")

    def test(
        self,
        test_data: DataGenerator,
        test_batch_size: int = settings.test_batch_size,
    ) -> None:
        X, Y_true = test_data[0]
        Y_pred = self.model.predict(X, verbose=1)
        for idx in range(test_batch_size):
            x = X[idx]
            y_true = Y_true[idx]
            y_pred = Y_pred[idx]

            img = utils.tensor_to_image(x)
            img = utils.draw_bounding_boxes(img, y_pred, pred=True)
            img = utils.draw_bounding_boxes(img, y_true)
            img.show()

    def predict(self, batch: tf.Tensor) -> tf.Tensor:
        return self.model.predict(batch, verbose=1)

    def _load_weights(self, path: str) -> None:
        self.model.load_weights(path)
