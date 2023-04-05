"""
Unit tests for the utils module.
"""

import numpy as np
import tensorflow as tf
from PIL import Image

from jaipy import settings, utils


def test_draw_bounding_boxes():
    zeros = np.zeros((7, 7, 5, 5))
    zeros[3, 3, :, 0] = np.array([1, 0.5, 0.5, 0.5, 0.5])
    tensor = tf.convert_to_tensor(zeros)

    image = Image.new("RGB", (settings.INPUT_SIZE, settings.INPUT_SIZE))
    image_true = utils.draw_bounding_boxes(image, tensor)
    image_pred = utils.draw_bounding_boxes(image, tensor, pred=True)

    assert image_true == image_pred
