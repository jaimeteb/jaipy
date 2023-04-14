"""
Mock data for testing.
"""

import typing as t

import numpy as np
import tensorflow as tf
from PIL import Image

from jaipy import dataset, settings


def get_mock_tensor_one_box_one_class() -> tf.Tensor:
    """
    tensor representing a prediction of a single
    bounding box in the center of the image
    """
    zeros = np.zeros((7, 7, 5, 5))
    zeros[3, 3, :, 0] = np.array([1.0, 0.5, 0.5, 0.5, 0.5])
    tensor = tf.convert_to_tensor(zeros)
    return tensor


def get_mock_tensor_one_box_one_class_v2() -> tf.Tensor:
    """
    tensor representing a prediction of a single
    bounding box in the center of the image
    """
    zeros = np.zeros((7, 7, 5, 5))
    zeros[2, 2, :, 0] = np.array([1.0, 0.5, 0.5, 4 / 14, 8 / 14])
    tensor = tf.convert_to_tensor(zeros)
    return tensor


def get_mock_tensor_one_box_one_class_v3() -> tf.Tensor:
    """
    tensor representing a prediction of a single
    bounding box in the center of the image
    """
    zeros = np.zeros((7, 7, 5, 5))
    zeros[2, 2, :, 0] = np.array([1.0, 0.4, 0.6, 5 / 14, 9 / 14])
    tensor = tf.convert_to_tensor(zeros)
    return tensor


def get_mock_tensor_two_boxes_one_class() -> tf.Tensor:
    """
    tensor representing a prediction of two
    bounding boxes for the same object class in/near
    the center of the image
    """
    zeros = np.zeros((7, 7, 5, 5))
    zeros[3, 3, :, 0] = np.array([1.0, 0.5, 0.5, 0.5, 0.5])
    zeros[4, 3, :, 0] = np.array([1.0, 0.4, 0.4, 0.4, 0.4])
    tensor = tf.convert_to_tensor(zeros)
    return tensor


def get_mock_tensor_two_boxes_different_classes() -> tf.Tensor:
    """
    tensor representing a prediction of two
    bounding boxes for different classes
    """
    zeros = np.zeros((7, 7, 5, 5))
    zeros[3, 2, :, 1] = np.array([1.0, 0.3, 0.3, 0.3, 0.3])
    zeros[4, 4, :, 2] = np.array([1.0, 0.6, 0.6, 0.6, 0.6])
    tensor = tf.convert_to_tensor(zeros)
    return tensor


def get_empty_image() -> Image.Image:
    return Image.new("RGB", (settings.INPUT_SIZE, settings.INPUT_SIZE))


def get_mock_annotations_image_category_indces() -> (
    t.Tuple[
        t.List[dataset.DatasetAnnotations],
        dataset.DatasetImages,
        t.Dict[int, int],
    ]
):
    """
    Mock annotations, image, and category indices.
    """
    return (
        [
            dataset.DatasetAnnotations(
                id=0,
                image_id=0,
                category_id=0,
                bbox=[3, 1, 4, 8],
                area=0.0,
                iscrowd=0,
                supercategory="",
            )
        ],
        dataset.DatasetImages(
            id=0,
            file_name="",
            height=14,
            width=14,
        ),
        {0: 0},
    )
