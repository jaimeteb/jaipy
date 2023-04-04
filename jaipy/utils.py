"""
Utility functions for jaipy.
"""


import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from jaipy import settings

COLORS = [
    "blue",
    "cyan",
    "green",
    "salmon",
    "yellow",
    "darkblue",
    "darkcyan",
    "darkgreen",
    "darksalmon",
    "darkyellow",
]


def tensor_to_image(tensor: tf.Tensor) -> Image.Image:
    return Image.fromarray(np.array(tensor * 255, dtype=np.uint8))


def draw_bounding_boxes(
    image: Image.Image, tensor: tf.Tensor, pred: bool = False
) -> Image.Image:
    bounding_boxes = tf.reshape(
        tensor,
        (tensor.shape[0] * tensor.shape[1], tensor.shape[2], tensor.shape[3]),
    )

    draw = ImageDraw.Draw(image)
    for bbidx in range(bounding_boxes.shape[0]):
        bbx = bounding_boxes[bbidx]
        for idx in range(bbx.shape[1]):
            if (bbx[0, idx] == 1 and not pred) or (
                bbx[0, idx] > settings.PREDICTION_THRESHOLD and pred
            ):
                x = bbx[1, idx] * image.width
                y = bbx[2, idx] * image.height
                w = bbx[3, idx] * image.width
                h = bbx[4, idx] * image.height
                box = ((x - w / 2, y - h / 2), (x + w / 2, y + h / 2))
                color = COLORS[idx] if not pred else COLORS[idx + 5]
                draw.rectangle(box, outline=color)

    return image
