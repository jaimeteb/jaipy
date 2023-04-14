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

    return image
