"""
Utility functions for jaipy.
"""
import typing as t

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


def get_obj_x_y_w_h(
    tensor: tf.Tensor,
) -> t.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Get object, x, y, w, h from tensor.
    """

    obj = tensor[..., 0]
    x = tensor[..., 1]
    y = tensor[..., 2]
    w = tensor[..., 3]
    h = tensor[..., 4]

    return obj, x, y, w, h


def draw_predictions(
    x: tf.Tensor,
    boxes: tf.Tensor,
    scores: tf.Tensor,
    classes: tf.Tensor,
    nums: tf.Tensor,
) -> t.List[Image.Image]:
    imgs = []
    for batch_idx in range(x.shape[0]):
        img = tensor_to_image(x[batch_idx])
        for idx in range(nums[batch_idx]):
            box = boxes[batch_idx, idx, ...] * img.width

            box1, box2 = (box[1], box[0]), (box[3], box[2])
            full_bbox = (box1, box2)

            draw = ImageDraw.Draw(img)
            draw.rectangle(
                full_bbox,
                outline=COLORS[int(classes[batch_idx, idx])],
            )
            draw.text(
                box1,
                f"{settings.classes[int(classes[batch_idx, idx])]}: {scores[batch_idx, idx]:.2f}",
                fill=COLORS[int(classes[batch_idx, idx])],
            )
        imgs.append(img)
    return imgs


def xywh_to_boxes(tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert bounding box coordinates from (x, y, w, h) to (y1, x1, y2, x2).
    """
    cx = tf.range(settings.grid)[:, None][None, :, None]
    cx = tf.broadcast_to(cx, [tensor.shape[0], settings.grid, settings.grid, 1])
    cx = tf.tile(cx, [1, 1, 1, settings.num_classes])
    cx = tf.cast(cx, tf.float32)

    cy = tf.range(settings.grid)[None, :, None]
    cy = tf.broadcast_to(cy, [tensor.shape[0], settings.grid, settings.grid, 1])
    cy = tf.tile(cy, [1, 1, 1, settings.num_classes])
    cy = tf.cast(cy, tf.float32)

    _, xp, yp, wp, hp = get_obj_x_y_w_h(tensor)
    xp = (xp + cx) / settings.grid
    yp = (yp + cy) / settings.grid

    x1 = xp - wp / 2
    y1 = yp - hp / 2
    x2 = xp + wp / 2
    y2 = yp + hp / 2

    boxes_pred = tf.stack([y1, x1, y2, x2], axis=-1)
    return boxes_pred
