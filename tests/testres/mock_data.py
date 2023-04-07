"""
Mock data for testing.
"""


import numpy as np
import tensorflow as tf
from PIL import Image

from jaipy import settings


def get_mock_tensor_one_box_one_class() -> tf.Tensor:
    """
    tensor representing a prediction of a single
    bounding box in the center of the image
    """
    zeros = np.zeros((7, 7, 5, 5))
    zeros[3, 3, :, 0] = np.array([1.0, 0.5, 0.5, 0.5, 0.5])
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
