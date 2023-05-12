"""
Convolutional Neural Network model definition
"""
# pyright: reportMissingImports=false

import datetime as dt
import json
import typing as t

import mlflow
import numpy as np
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

    def predict(self, X: tf.Tensor, nms: bool = False, show: bool = True):
        with utils.device:
            if not nms:
                return self.model.predict(X, verbose=1)

            Y_pred = self.model.predict(X, verbose=1).astype("float32")

            boxes, scores, classes, nums = _boxes_scores_classes_nums(Y_pred)

            imgs = utils.draw_predictions(X, boxes, scores, classes, nums)
            if show:
                _ = [img.show() for img in imgs]
                return None
            return imgs

    def predict_and_evaluate(
        self,
        X: tf.Tensor,
        Y_true: tf.Tensor,
    ):
        with utils.device:
            Y_pred = self.model.predict(X, verbose=1).astype("float32")
            Y_true = Y_true.numpy().astype("float32")

            boxes_p, scores_p, classes_p, nums_p = _boxes_scores_classes_nums(Y_pred)
            boxes_t, scores_t, classes_t, nums_t = _boxes_scores_classes_nums(
                Y_true, nms=False
            )

            # imgs_p = utils.draw_predictions(X, boxes_p, scores_p, classes_p, nums_p)
            # imgs_t = utils.draw_predictions(X, boxes_t, scores_t, classes_t, nums_t)
            # for img_p, img_t in zip(imgs_p, imgs_t):
            #     img_p.show()
            #     img_t.show()

            average_precisions = []
            average_precisions_dict = {}
            for img_idx in range(X.shape[0]):
                valid_p = nums_p.numpy()[img_idx]
                valid_t = nums_t.numpy()[img_idx]

                bp = boxes_p[img_idx, :valid_p, :]
                sp = scores_p[img_idx, :valid_p]
                cp = classes_p[img_idx, :valid_p]

                bt = boxes_t[img_idx, :valid_t, :]
                st = scores_t[img_idx, :valid_t]
                ct = classes_t[img_idx, :valid_t]

                for class_idx in np.unique(classes_t):
                    _bp = bp[cp == class_idx]
                    _sp = sp[cp == class_idx]
                    _bt = bt[ct == class_idx]
                    _st = st[ct == class_idx]

                    tps = np.zeros(valid_p)
                    fps = np.zeros(valid_p)

                    for idx, (box_p, _) in enumerate(zip(_bp, _sp)):
                        ious = utils.iou(box_p, _bt)
                        ious = tf.where(ious > settings.iou_threshold, 1, 0)
                        if tf.reduce_sum(ious) == 0:
                            fps[idx] = 1
                        else:
                            tps[idx] = 1
                    tps = np.cumsum(tps)
                    fps = np.cumsum(fps)
                    precision = tps / (tps + fps + 1e-8)
                    recall = tps / (valid_t + 1e-8)
                    ap = utils.average_precision(precision, recall)
                    average_precisions.append(ap)
                    average_precisions_dict.setdefault(class_idx, []).append(ap)

            mean_average_precision = np.mean(average_precisions)
            logger.info("mAP: %s", mean_average_precision)

            for class_idx, aps in average_precisions_dict.items():
                logger.info("class: %s, AP: %s", class_idx, np.mean(aps))

    def load_weights(self, path: str) -> None:
        self.model.load_weights(path)


def _boxes_scores_classes_nums(
    Y_pred: tf.Tensor,
    nms: bool = True,
) -> t.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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
    if nms:
        boxes, scores, classes, nums = tf.image.combined_non_max_suppression(
            boxes_tensor,
            scores_tensor,
            max_output_size_per_class=settings.grid**2,
            max_total_size=settings.grid**2,
            iou_threshold=settings.iou_threshold,
            score_threshold=settings.prediction_threshold,
        )
    else:
        all_boxes = []
        all_scores = []
        all_classes = []
        all_nums = []
        for img_idx in range(Y_pred.shape[0]):
            boxes = boxes_pred[img_idx][obj_pred[img_idx] == 1]
            scores = obj_pred[img_idx][obj_pred[img_idx] == 1]
            nums = len(boxes)

            mask = tf.reduce_any(scores_tensor[img_idx] != [0, 0, 0, 0, 0], axis=-1)
            classes = tf.argmax(scores_tensor[img_idx], axis=-1)
            classes = tf.gather_nd(classes, tf.where(mask))

            # fill boxes
            input_tensor = boxes
            output_shape = (49, 4)
            num_rows_to_add_end = output_shape[0] - tf.shape(input_tensor)[0]
            paddings = tf.stack([[0, num_rows_to_add_end], [0, 0]])
            boxes = tf.pad(input_tensor, paddings, "CONSTANT")

            # fill scores
            input_tensor = scores
            output_shape = (49,)
            num_rows_to_add_end = output_shape[0] - tf.shape(input_tensor)[0]
            paddings = tf.stack([[0, num_rows_to_add_end]])
            scores = tf.pad(input_tensor, paddings, "CONSTANT")

            # fill classes
            input_tensor = classes
            output_shape = (49,)
            num_rows_to_add_end = output_shape[0] - tf.shape(input_tensor)[0]
            paddings = tf.stack([[0, num_rows_to_add_end]])
            classes = tf.pad(input_tensor, paddings, "CONSTANT")

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)
            all_nums.append(nums)

        boxes = tf.stack(all_boxes)
        scores = tf.stack(all_scores)
        classes = tf.stack(all_classes)
        nums = tf.stack(all_nums)

    return boxes, scores, classes, nums
