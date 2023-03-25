"""
Image dataset loading and parsing
"""
# pyright: reportMissingImports=false

import os
import random
import typing as t

import numpy as np
import tensorflow as tf
from pydantic import (  # parse_obj_as,; pylint: disable=no-name-in-module
    BaseModel,
    parse_file_as,
)
from tensorflow.keras.utils import img_to_array, load_img

from jaipy import settings
from jaipy.logger import logger


class DatasetCategories(BaseModel):
    id: int
    name: str
    supercategory: str


class DatasetImages(BaseModel):
    id: int
    file_name: str
    height: int
    width: int


class DatasetAnnotations(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: t.List[int]
    area: float
    iscrowd: int
    supercategory: str


class DatasetLabels(BaseModel):
    info: dict
    categories: t.List[DatasetCategories]
    images: t.List[DatasetImages]
    annotations: t.List[DatasetAnnotations]


def generate_dataset_files():
    import fiftyone as fo  # pylint: disable=import-outside-toplevel
    import fiftyone.zoo as foz  # pylint: disable=import-outside-toplevel

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        splits=["train", "test", "validation"],
        label_types=["detections"],
        classes=settings.CLASSES,
    )
    dataset.export(settings.EXPORT_DIR, fo.types.COCODetectionDataset)
    dataset.delete()


def get_dataset_labels() -> DatasetLabels:
    logger.info("Loading dataset labels")
    labels = parse_file_as(
        DatasetLabels, os.path.join(settings.EXPORT_DIR, "labels.json")
    )

    return labels


def get_images(n_images: int = -1) -> t.Tuple[tf.Tensor, tf.Tensor]:
    labels = get_dataset_labels()

    def _get_image_annotations() -> t.Dict[int, t.List[DatasetAnnotations]]:
        nonlocal labels
        image_annotations: t.Dict[int, t.List[DatasetAnnotations]] = {}
        for ann in labels.annotations:
            image_annotations.setdefault(ann.image_id, []).append(ann)

        return image_annotations

    category_indices = {}
    idx = 0
    for cat in labels.categories:
        if cat.name in settings.CLASSES:
            category_indices[cat.id] = idx
            idx += 1

    image_annotations = _get_image_annotations()
    images_sample = (
        random.sample(labels.images, k=n_images) if n_images > 0 else labels.images
    )

    logger.info("Creating image and data tensors")
    X = []
    Y = []
    for img in images_sample:
        img_data = load_img(os.path.join(settings.EXPORT_DIR, "data", img.file_name))
        img_data = tf.image.resize(img_data, (settings.INPUT_SIZE, settings.INPUT_SIZE))
        X.append(img_to_array(img_data) / 255)

        Y_temp = np.zeros(
            # (settings.GRID, settings.GRID, 5, settings.NUM_CLASSES + settings.NUM_BOXES)
            (
                settings.GRID,
                settings.GRID,
                5,
                settings.NUM_CLASSES,
            )
        )

        if img.id in image_annotations:
            for ann in image_annotations[img.id]:
                if ann.category_id in category_indices:
                    x = ann.bbox[0] + ann.bbox[2] / 2
                    y = ann.bbox[1] + ann.bbox[3] / 2
                    w = ann.bbox[2]
                    h = ann.bbox[3]
                    x /= img.width
                    y /= img.height
                    w /= img.width
                    h /= img.height

                    grid_x = int(x * settings.GRID)
                    grid_y = int(y * settings.GRID)

                    class_index = category_indices[ann.category_id]

                    Y_temp[grid_x, grid_y, 0, class_index] = 1
                    Y_temp[grid_x, grid_y, 1, class_index] = x
                    Y_temp[grid_x, grid_y, 2, class_index] = y
                    Y_temp[grid_x, grid_y, 3, class_index] = w
                    Y_temp[grid_x, grid_y, 4, class_index] = h

        Y.append(tf.convert_to_tensor(Y_temp))

    X_tensor = tf.stack(X)
    Y_tensor = tf.stack(Y)
    return X_tensor, Y_tensor
