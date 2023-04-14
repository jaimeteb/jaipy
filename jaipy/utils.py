"""
Utility functions for jaipy.
"""


import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

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


def tensor_to_image(tensor: tf.Tensor) -> Image.Image:
    return Image.fromarray(np.array(tensor * 255, dtype=np.uint8))


def draw_bounding_boxes(
    image: Image.Image, tensor: tf.Tensor, pred: bool = False
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for ci in range(tensor.shape[0]):
        for cj in range(tensor.shape[1]):
            for idx in range(tensor.shape[3]):
                if (tensor[ci, cj, 0, idx] == 1 and not pred) or (
                    tensor[ci, cj, 0, idx] > settings.prediction_threshold and pred
                ):
                    x = (tensor[ci, cj, 1, idx] + ci) * int(image.width / settings.grid)
                    y = (tensor[ci, cj, 2, idx] + cj) * int(image.width / settings.grid)
                    w = tensor[ci, cj, 3, idx] * image.width
                    h = tensor[ci, cj, 4, idx] * image.height
                    box = ((x - w / 2, y - h / 2), (x + w / 2, y + h / 2))
                    color = COLORS[idx] if not pred else COLORS[idx + 5]
                    draw.rectangle(box, outline=color)

    return image
