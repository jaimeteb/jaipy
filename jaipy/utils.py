"""
Utility functions for jaipy.
"""


import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from jaipy.logger import logger
from jaipy.settings import settings

COLORS = [
    "blue",
    "cyan",
    "green",
    "salmon",
    "orange",
    "darkblue",
    "darkcyan",
    "darkgreen",
    "darksalmon",
    "darkorange",
]

is_gpu = tf.config.list_physical_devices("GPU")
if is_gpu:
    logger.info("GPU available.")
    device = tf.device("/gpu:0")
else:
    logger.info("GPU not available.")
    device = tf.device("/cpu:0")


def tensor_to_image(tensor: tf.Tensor) -> Image.Image:
    return Image.fromarray(np.array(tensor * 255, dtype=np.uint8))


def draw_bounding_boxes(
    image: Image.Image, tensor: tf.Tensor, pred: bool = False
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for ci in range(tensor.shape[0]):
        for cj in range(tensor.shape[1]):
            for idx in range(tensor.shape[2]):
                if (tensor[ci, cj, idx, 0] == 1 and not pred) or (
                    tensor[ci, cj, idx, 0] > settings.prediction_threshold and pred
                ):
                    x = (tensor[ci, cj, idx, 1] + ci) * int(image.width / settings.grid)
                    y = (tensor[ci, cj, idx, 2] + cj) * int(image.width / settings.grid)
                    w = tensor[ci, cj, idx, 3] * image.width
                    h = tensor[ci, cj, idx, 4] * image.height
                    box = ((x - w / 2, y - h / 2), (x + w / 2, y + h / 2))

                    color = COLORS[idx] if not pred else COLORS[idx + 5]
                    draw.rectangle(box, outline=color)
                    text = (
                        f"{settings.classes[idx]}"
                        if not pred
                        else f"{settings.classes[idx]}: {tensor[ci, cj, idx, 0]:.2f}"
                    )
                    draw.text((x - w / 2, y - h / 2), text, fill=color)

    return image


def draw_prediction_and_truth(
    x: tf.Tensor, y_pred: tf.Tensor, y_true: tf.Tensor
) -> Image.Image:
    img = tensor_to_image(x)
    img = draw_bounding_boxes(img, y_pred, pred=True)
    img = draw_bounding_boxes(img, y_true)
    return img
