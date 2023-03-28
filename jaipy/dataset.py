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
from tensorflow.keras.utils import Sequence, img_to_array, load_img

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


class DataGenerator(Sequence):
    def __init__(
        self,
        batch_size: int = 32,
        # cutoff_start: float = 0.0,
        # cutoff_end: float = 1.0,
    ):
        self.batch_size: int = batch_size

        self.labels: DatasetLabels = get_dataset_labels()
        self.category_indices: t.Dict[int, int] = self._get_category_indices()

        self.images_dict: t.Dict[int, DatasetImages] = {
            img.id: img for img in self.labels.images
        }
        self.image_annotations_dict: t.Dict[
            int, t.List[DatasetAnnotations]
        ] = self._get_image_annotations()

        self._shuffle_indices()

    def _get_category_indices(self) -> t.Dict[int, int]:
        category_indices: t.Dict[int, int] = {}
        idx = 0
        for cat in self.labels.categories:
            if cat.name in settings.CLASSES:
                category_indices[cat.id] = idx
                idx += 1
        return category_indices

    def _get_image_annotations(  # pylint: disable=fixme
        self,
    ) -> t.Dict[int, t.List[DatasetAnnotations]]:
        image_annotations: t.Dict[int, t.List[DatasetAnnotations]] = {}
        # TODO: apply cutoffs here
        for ann in self.labels.annotations:
            if self.category_indices.get(ann.category_id) is not None:
                image_annotations.setdefault(ann.image_id, []).append(ann)

        return image_annotations

    def _shuffle_indices(self):
        self.indices = list(self.image_annotations_dict.keys()).copy()
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.image_annotations_dict) // self.batch_size

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X = []
        Y = []
        for idx in indices:
            img = self.images_dict[idx]
            img_data = load_img(
                os.path.join(settings.EXPORT_DIR, "data", img.file_name)
            )
            img_data = tf.image.resize(
                img_data, (settings.INPUT_SIZE, settings.INPUT_SIZE)
            )
            X.append(img_to_array(img_data) / 255)

            # (settings.GRID, settings.GRID, 5, settings.NUM_CLASSES + settings.NUM_BOXES)
            y_shape = (settings.GRID, settings.GRID, 5, settings.NUM_CLASSES)
            Y_temp = np.zeros(y_shape)

            # if img.id in self.image_annotations_dict:
            for ann in self.image_annotations_dict[img.id]:
                if ann.category_id in self.category_indices:
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

                    class_index = self.category_indices[ann.category_id]

                    Y_temp[grid_x, grid_y, 0, class_index] = 1
                    Y_temp[grid_x, grid_y, 1, class_index] = x
                    Y_temp[grid_x, grid_y, 2, class_index] = y
                    Y_temp[grid_x, grid_y, 3, class_index] = w
                    Y_temp[grid_x, grid_y, 4, class_index] = h

            Y.append(tf.convert_to_tensor(Y_temp))

        X_tensor = tf.stack(X)
        Y_tensor = tf.stack(Y)
        return X_tensor, Y_tensor

    def on_epoch_end(self):
        self._shuffle_indices()
