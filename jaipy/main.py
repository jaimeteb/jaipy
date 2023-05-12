# pylint: disable=no-member

"""
Main module
"""

import glob
import time

from jaipy.dataset import (
    DataGenerator,
    convert_cv2_image_to_yolo_like_tensor,
    generate_dataset_files,
    generate_test_dataset_files,
)
from jaipy.model import Model
from jaipy.settings import settings


def train():
    model = Model()
    model.train(
        train_data=DataGenerator(
            batch_size=settings.batch_size,
            cutoff_start=settings.train_cutoff_start,
            cutoff_end=settings.train_cutoff_end,
        ),
        val_data=DataGenerator(
            batch_size=settings.batch_size,
            cutoff_start=settings.val_cutoff_start,
            cutoff_end=settings.val_cutoff_end,
            shuffle=False,
        ),
    )


def train_test():
    model = Model()
    model.train(
        train_data=DataGenerator(
            batch_size=settings.batch_size,
            cutoff_start=settings.train_cutoff_start,
            cutoff_end=settings.train_cutoff_end,
        ),
        val_data=DataGenerator(
            batch_size=settings.batch_size,
            cutoff_start=settings.val_cutoff_start,
            cutoff_end=settings.val_cutoff_end,
            shuffle=False,
        ),
    )
    model.test(
        test_data=DataGenerator(
            batch_size=settings.test_batch_size,
            cutoff_start=settings.test_cutoff_start,
            cutoff_end=settings.test_cutoff_end,
            shuffle=False,
        )
    )


def test():
    if settings.weights_file is not None:
        model = Model()
        model.load_weights(settings.weights_file)
        model.test(
            test_data=DataGenerator(
                batch_size=settings.test_batch_size,
                cutoff_start=settings.test_cutoff_start,
                cutoff_end=settings.test_cutoff_end,
                shuffle=False,
            )
        )


def predict():
    if settings.weights_file is not None:
        model = Model()
        model.load_weights(settings.weights_file)

        test_data = DataGenerator(
            batch_size=settings.test_batch_size,
            cutoff_start=settings.test_cutoff_start,
            cutoff_end=settings.test_cutoff_end,
            shuffle=False,
        )
        X, _ = test_data[0]
        model.predict(X, nms=True)


def evaluate():
    if settings.weights_file is not None:
        model = Model()
        model.load_weights(settings.weights_file)

        test_data = DataGenerator(
            batch_size=settings.test_batch_size,
            cutoff_start=settings.test_cutoff_start,
            cutoff_end=settings.test_cutoff_end,
            shuffle=False,
            evaluate=True,
        )
        X, Y_true = test_data[0]
        model.predict_and_evaluate(X, Y_true)


def live_predict():
    if settings.weights_file is not None:
        import cv2  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        model = Model()
        model.load_weights(settings.weights_file)

        cap = cv2.VideoCapture(0)
        total_time = 0.0
        num_predictions = 0
        while True:
            ret, image = cap.read()
            if not ret:
                break

            start_time = time.monotonic()
            image = convert_cv2_image_to_yolo_like_tensor(image)
            image_pred = model.predict(image, nms=True, show=False)[0]  # type: ignore
            end_time = time.monotonic()

            total_time += end_time - start_time
            num_predictions += 1

            cv2.imshow(
                "Object Detection",
                cv2.cvtColor(np.array(image_pred), cv2.COLOR_BGR2RGB),
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        avg_time = total_time / num_predictions
        print(f"Number of predictions: {num_predictions}")
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Average time per prediction: {avg_time:.3f} seconds")


def predict_images():
    if settings.weights_file is not None:
        import cv2  # pylint: disable=import-outside-toplevel

        model = Model()
        model.load_weights(settings.weights_file)

        for image_file in glob.glob(f"{settings.images_dir}/*.jpg"):
            image = cv2.imread(image_file)
            image = convert_cv2_image_to_yolo_like_tensor(image)
            model.predict(image, nms=True, show=True)


def train_test_mock():
    model = Model(batch_size=1)

    dg = DataGenerator(test=True)
    model.train(train_data=dg, val_data=dg, checkpoints=False)
    model.test(test_data=dg, test_batch_size=1)


def generate_dataset():
    generate_dataset_files()


def generate_test_dataset():
    generate_test_dataset_files()
