"""
Convolutional Neural Network model definition
"""
# pyright: reportMissingImports=false

import datetime as dt
import typing as t

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Input,
    LeakyReLU,
    MaxPool2D,
    Reshape,
)
from tensorflow.keras.regularizers import l2

from jaipy import settings
from jaipy.dataset import DataGenerator
from jaipy.logger import logger


def Convolutional(
    layer: t.Any,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    batch_norm: bool = True,
):
    padding = "same"
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


class Model:
    input_size: int = settings.INPUT_SIZE
    channels: int = settings.CHANNELS
    num_classes: int = settings.NUM_CLASSES
    num_boxes: int = settings.NUM_BOXES

    def __init__(self):
        self.model = self._create_model()

    def _create_model(self) -> tf.keras.Model:
        input_layer = Input([self.input_size, self.input_size, self.channels])

        # Darknet-like architecture
        layer = Convolutional(input_layer, 32, 3)
        layer = MaxPool2D(2, 2, "same")(layer)

        layer = Convolutional(layer, 64, 3)
        layer = MaxPool2D(2, 2, "same")(layer)

        layer = Convolutional(layer, 128, 3)
        layer = Convolutional(layer, 64, 1)
        layer = Convolutional(layer, 128, 3)
        layer = MaxPool2D(2, 2, "same")(layer)

        layer = Convolutional(layer, 256, 3)
        layer = Convolutional(layer, 128, 1)
        layer = Convolutional(layer, 256, 3)
        layer = MaxPool2D(2, 2, "same")(layer)

        layer = Convolutional(layer, 512, 3)
        layer = Convolutional(layer, 256, 1)
        layer = Convolutional(layer, 512, 3)
        layer = Convolutional(layer, 256, 1)
        layer = Convolutional(layer, 512, 3)
        layer = MaxPool2D(2, 2, "same")(layer)

        layer = Convolutional(layer, 1024, 3)
        layer = Convolutional(layer, 512, 1)
        layer = Convolutional(layer, 1024, 3)
        layer = Convolutional(layer, 512, 1)
        layer = Convolutional(layer, 1024, 3)

        # Head
        layer = Conv2D(
            # filters=5 * (self.num_classes + self.num_boxes),
            filters=5 * (self.num_classes),
            kernel_size=(1, 1),
            strides=1,
        )(layer)
        layer = Activation("linear", name="linear_activation")(layer)

        shape = layer.shape
        final_layer = Reshape(
            # (shape[1], shape[2], 5, self.num_classes + self.num_boxes),
            (shape[1], shape[2], 5, self.num_classes),
        )(layer)

        # Create model
        model = tf.keras.Model(input_layer, final_layer)
        model.summary(print_fn=logger.info)

        return model

    def train(
        self,
        # x_train: tf.Tensor,
        # y_train: tf.Tensor,
    ) -> None:
        # assert x_train.shape[0] == y_train.shape[0]

        # cutoff = int(x_train.shape[0] * 0.9)
        # x_train_split, x_val_split = x_train[:cutoff], x_train[cutoff:]
        # y_train_split, y_val_split = y_train[:cutoff], y_train[cutoff:]

        data_generator = DataGenerator(batch_size=64)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.000_1),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError()],
        )
        self.model.fit(
            data_generator,
            # x_train_split,
            # y_train_split,
            # validation_data=(x_val_split, y_val_split),
            batch_size=64,
            epochs=10,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=f"./logs/{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    histogram_freq=1,
                ),
            ],
        )

    def predict(self, batch: tf.Tensor) -> tf.Tensor:
        return self.model.predict(batch, verbose=0)
