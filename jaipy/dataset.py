"""
Image dataset loading and parsing
"""
# pyright: reportMissingImports=false

import os
import random
import typing as t

import tensorflow as tf
from pydantic import BaseModel, parse_file_as  # pylint: disable=no-name-in-module
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


# def get_images(n_images: int) -> t.Tuple[t.List, t.List]:
def get_images(n_images: int) -> tf.Tensor:
    labels = get_dataset_labels()
    images_sample = random.sample(labels.images, k=n_images)

    X = []

    images_sample_annotations: t.Dict[int, t.List[DatasetAnnotations]] = {}
    for idx, img in enumerate(images_sample):
        if idx % 100 == 0:
            logger.info("Loading %d images", idx)
        for ann in labels.annotations:
            if ann.image_id == img.id:
                images_sample_annotations.setdefault(idx, []).append(ann)

        img_data = load_img(os.path.join(settings.EXPORT_DIR, "data", img.file_name))
        img_data = tf.image.resize(img_data, (settings.INPUT_SIZE, settings.INPUT_SIZE))
        X.append(img_to_array(img_data))

    logger.info(images_sample_annotations)
    # print(X)
    X_tensor = tf.stack(X)
    return X_tensor
