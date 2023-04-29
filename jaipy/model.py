"""
Convolutional Neural Network model definition
"""
# pyright: reportMissingImports=false

import datetime as dt
import json
import typing as t

import mlflow
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
from jaipy.callbacks import ImagePredictionCallback, MLFlowCallback
from jaipy.dataset import DataGenerator
from jaipy.logger import logger
from jaipy.settings import settings

tf.keras.backend.set_floatx("float64")


def Convolutional(  # pylint: disable=too-many-arguments
    layer: t.Any,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    padding: str = "same",
):
    layer = Conv2D(
        activation=None,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_regularizer=l2(0.000_5),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0),
    )(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.1)(layer)

    return layer


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
        file_name = settings.architecture_file
        with open(file_name, "r", encoding="utf-8") as file:
            architecture = json.load(file)

        input_layer = Input([self.input_size, self.input_size, self.channels])

        layer = input_layer
        for spec in architecture.get("darknet", [{}]):
            if spec.get("type") == "convolutional":
                layer = Convolutional(
                    layer,
                    spec.get("filters"),
                    spec.get("kernel_size"),
                    spec.get("strides", 1),
                    spec.get("padding", "same"),
                )
            elif spec.get("type") == "maxpool":
                layer = MaxPool2D(
                    spec.get("pool_size"),
                    spec.get("strides"),
                    spec.get("padding", "same"),
                )(layer)
        layer = Flatten()(layer)
        for spec in architecture.get("fully_connected", [{}]):
            if spec.get("type") == "dense":
                layer = Dense(
                    spec.get("units")
                    if not spec.get("final", False)
                    else self.grid**2 * 5 * self.num_classes,
                    activation=LeakyReLU(alpha=0.1)
                    if spec.get("activation", "leaky_relu") == "leaky_relu"
                    else "linear",
                )(layer)
            elif spec.get("type") == "dropout":
                layer = Dropout(spec.get("rate"))(layer)

        layer = Reshape((self.grid, self.grid, 5, self.num_classes))(layer)

        model = tf.keras.Model(input_layer, layer)
        # model.summary(print_fn=logger.info)
        logger.info("Created model according to architecture file %s", file_name)

        return model

    def train(
        self,
        train_data: DataGenerator,
        val_data: DataGenerator,
        checkpoints: bool = True,
    ) -> None:
        model_name = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.info("Starting training for %s", model_name)
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params=settings.dict())
            mlflow.run_name = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            with utils.device:
                yolo_like_loss = loss.YOLOLikeLoss()
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.learning_rate
                    ),
                    # loss=tf.keras.losses.MeanSquaredError(),
                    loss=yolo_like_loss,
                )

                callbacks = [
                    tf.keras.callbacks.TensorBoard(
                        log_dir=f"./logs/{model_name}",
                        histogram_freq=1,
                    ),
                ]

                if checkpoints:
                    callbacks.extend(
                        [
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath=f"./models/{model_name}" + "-{epoch:02d}.h5",
                                save_best_only=False,
                                save_weights_only=True,
                            ),
                            ImagePredictionCallback(
                                model_name=model_name,
                                test_data=val_data,
                                test_batch_size=settings.test_batch_size,
                            ),
                            MLFlowCallback(),
                        ]
                    )
                if settings.checkpoint_file is not None:
                    logger.info("Loading weights from %s", settings.checkpoint_file)
                    self.model.load_weights(settings.checkpoint_file)

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
        logger.info("Starting testing")
        with utils.device:
            X, Y_true = test_data[0]
            Y_pred = self.model.predict(X, verbose=1)
        for idx in range(test_batch_size):
            img = utils.draw_prediction_and_truth(X[idx], Y_pred[idx], Y_true[idx])
            img.show()

    def predict(self, X: tf.Tensor, nms: bool = False) -> t.Optional[tf.Tensor]:
        with utils.device:
            if not nms:
                return self.model.predict(X, verbose=1)

            Y_pred = self.model.predict(X, verbose=1).astype("float32")

            obj_pred = Y_pred[..., 0]
            boxes_pred = utils.xywh_to_boxes(Y_pred)

            boxes_tensor = tf.reshape(
                boxes_pred,
                shape=(-1, settings.grid * settings.grid, settings.num_classes, 4),
            )
            scores_tensor = tf.reshape(
                obj_pred,
                shape=(-1, settings.grid * settings.grid, settings.num_classes),
            )
            boxes, scores, classes, nums = tf.image.combined_non_max_suppression(
                boxes_tensor,
                scores_tensor,
                max_output_size_per_class=settings.grid**2,
                max_total_size=settings.grid**2,
                iou_threshold=0.5,
                score_threshold=settings.prediction_threshold,
            )

            imgs = utils.draw_predictions(X, boxes, scores, classes, nums)
            _ = [img.show() for img in imgs]
            return None

    def load_weights(self, path: str) -> None:
        self.model.load_weights(path)
