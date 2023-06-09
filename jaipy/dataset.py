# pylint: disable=no-member

"""
Image dataset loading and parsing
"""
# pyright: reportMissingImports=false

import functools
import os
import random
import typing as t

import cv2
import numpy as np
import tensorflow as tf
from pydantic import (  # parse_obj_as,; pylint: disable=no-name-in-module
    BaseModel,
    parse_file_as,
)
from tensorflow.keras.utils import Sequence, img_to_array, load_img

from jaipy.logger import logger
from jaipy.settings import settings


class DatasetCategories(BaseModel):
    id: int
    name: str
    supercategory: t.Optional[str]


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
    supercategory: t.Optional[str]


class DatasetLabels(BaseModel):
    info: dict
    categories: t.List[DatasetCategories]
    images: t.List[DatasetImages]
    annotations: t.List[DatasetAnnotations]


def generate_dataset_files():
    import fiftyone as fo  # pylint: disable=import-outside-toplevel
    import fiftyone.zoo as foz  # pylint: disable=import-outside-toplevel

    dataset = foz.load_zoo_dataset(
        settings.dataset_name,
        splits=["train", "test", "validation"],
        label_types=["detections"],
        classes=settings.classes,
    )
    dataset.export(settings.export_dir, fo.types.COCODetectionDataset)
    dataset.delete()


def generate_test_dataset_files():
    import fiftyone as fo  # pylint: disable=import-outside-toplevel
    import fiftyone.zoo as foz  # pylint: disable=import-outside-toplevel

    dataset = foz.load_zoo_dataset(
        settings.test_dataset_name,
        splits=["train", "validation"],
        label_types=["detections"],
        classes=settings.classes,
    )
    dataset.export(settings.test_export_dir, fo.types.COCODetectionDataset)
    dataset.delete()


@functools.lru_cache(maxsize=1)
def get_dataset_labels(data_dir: str) -> DatasetLabels:
    file_name = os.path.join(data_dir, "labels.json")
    if not os.path.exists(file_name):
        logger.warning("Generating dataset labels")
        generate_dataset_files()
    logger.info("Loading dataset labels")
    labels = parse_file_as(
        DatasetLabels,
        file_name,
    )

    return labels


class DataGenerator(Sequence):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        batch_size: int = 32,
        cutoff_start: float = 0.0,
        cutoff_end: float = 1.0,
        test: bool = False,
        shuffle: bool = True,
        evaluate: bool = False,
    ):
        self.batch_size: int = batch_size if not test else 1
        if test:
            self.data_dir: str = "./tests/testres"
        elif evaluate:
            self.data_dir: str = settings.test_export_dir
        else:
            self.data_dir: str = settings.export_dir
        self.labels: DatasetLabels = get_dataset_labels(self.data_dir)
        self.category_indices: t.Dict[int, int] = self._get_category_indices()

        self.images_dict: t.Dict[int, DatasetImages] = {
            img.id: img for img in self.labels.images
        }
        self.image_annotations_dict: t.Dict[
            int, t.List[DatasetAnnotations]
        ] = self._get_image_annotations(cutoff_start, cutoff_end)

        logger.info("Loaded %s images", len(self.image_annotations_dict))
        self._shuffle_indices(shuffle)

    def _get_category_indices(self) -> t.Dict[int, int]:
        return {
            cat.id: settings.classes.index(cat.name)
            for cat in self.labels.categories
            if cat.name in settings.classes
        }

    def _get_image_annotations(
        self, cutoff_start: float, cutoff_end: float
    ) -> t.Dict[int, t.List[DatasetAnnotations]]:
        image_annotations: t.Dict[int, t.List[DatasetAnnotations]] = {}

        total_annotations = len(self.labels.annotations)
        cutoff_start_index = int(total_annotations * cutoff_start)
        cutoff_end_index = int(total_annotations * cutoff_end) - 1

        for idx, ann in enumerate(self.labels.annotations):
            if (
                self.category_indices.get(ann.category_id) is not None
                and cutoff_start_index <= idx <= cutoff_end_index
            ):
                image_annotations.setdefault(ann.image_id, []).append(ann)

        return image_annotations

    def _shuffle_indices(self, shuffle: bool = True):
        self.indices = list(self.image_annotations_dict.keys()).copy()
        if shuffle:
            random.shuffle(self.indices)

    def __len__(self):
        return len(self.image_annotations_dict) // self.batch_size

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X = []
        Y = []
        for idx in indices:
            img = self.images_dict[idx]

            X.append(
                tf.convert_to_tensor(
                    convert_image_to_yolo_like_tensor(
                        data_dir=self.data_dir,
                        img=img,
                    )
                )
            )
            Y.append(
                tf.convert_to_tensor(
                    convert_annotations_to_yolo_like_tensor(
                        anns=self.image_annotations_dict[img.id],
                        img=img,
                        category_indices=self.category_indices,
                    )
                )
            )

        X_tensor = tf.stack(X)
        Y_tensor = tf.stack(Y)
        return X_tensor, Y_tensor

    def on_epoch_end(self):
        self._shuffle_indices()


def convert_image_to_yolo_like_tensor(
    data_dir: str,
    img: DatasetImages,
) -> tf.Tensor:
    img_data = load_img(os.path.join(data_dir, "data", img.file_name))
    img_data = tf.image.resize(img_data, (settings.input_size, settings.input_size))
    return tf.convert_to_tensor(img_to_array(img_data) / 255)


def convert_cv2_image_to_yolo_like_tensor(
    img: np.ndarray,
) -> tf.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (settings.input_size, settings.input_size))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return tf.convert_to_tensor(img)


def convert_annotations_to_yolo_like_tensor(
    anns: t.List[DatasetAnnotations],
    img: DatasetImages,
    category_indices: t.Dict[int, int],
) -> tf.Tensor:
    y_shape = (settings.grid, settings.grid, 5, settings.num_classes)
    Y_temp = np.zeros(y_shape)

    for ann in anns:
        if ann.category_id in category_indices:
            x = (ann.bbox[0] + ann.bbox[2] / 2) / img.width
            y = (ann.bbox[1] + ann.bbox[3] / 2) / img.height
            w = (ann.bbox[2]) / img.width
            h = (ann.bbox[3]) / img.height

            grid_x = int(x * settings.grid)
            grid_y = int(y * settings.grid)

            x = x * settings.grid - grid_x
            y = y * settings.grid - grid_y

            class_index = category_indices[ann.category_id]

            Y_temp[grid_x, grid_y, class_index, 0] = 1
            Y_temp[grid_x, grid_y, class_index, 1] = x
            Y_temp[grid_x, grid_y, class_index, 2] = y
            Y_temp[grid_x, grid_y, class_index, 3] = w
            Y_temp[grid_x, grid_y, class_index, 4] = h

    return tf.convert_to_tensor(Y_temp)
