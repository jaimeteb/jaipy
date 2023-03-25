"""
Image dataset loading and parsing
"""
# pyright: reportMissingImports=false

import os
import random
import typing as t

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


# def write_image_annotations():
#     labels = get_dataset_labels()
#     total_annotations = len(labels.annotations)

#     logger.info("Generating annotations.csv")
#     images_dict: t.Dict[int, DatasetImages] = {img.id: img for img in labels.images}
#     with open("annotations.csv", "w+") as f:
#         for ann in labels.annotations:
#             img = images_dict[ann.image_id]
#             f.write(
#                 f"{img.id},{img.file_name},{img.height},{img.width},"
#                 f"{ann.category_id},{ann.bbox[0]},{ann.bbox[1]},{ann.bbox[2]},{ann.bbox[3]}\n"
#             )


def get_images(n_images: int) -> t.Tuple[tf.Tensor, tf.Tensor]:
    labels = get_dataset_labels()

    def _get_image_annotations() -> t.Dict[int, t.List[DatasetAnnotations]]:
        nonlocal labels
        image_annotations: t.Dict[int, t.List[DatasetAnnotations]] = {}
        for ann in labels.annotations:
            image_annotations.setdefault(ann.image_id, []).append(ann)

        return image_annotations

    image_annotations = _get_image_annotations()
    images_sample = random.sample(labels.images, k=n_images)

    logger.info("Creating image and data tensors")
    X = []
    Y = []
    for img in images_sample:
        img_data = load_img(os.path.join(settings.EXPORT_DIR, "data", img.file_name))
        img_data = tf.image.resize(img_data, (settings.INPUT_SIZE, settings.INPUT_SIZE))
        X.append(img_to_array(img_data))

        if img.id in image_annotations:
            Y.append(
                [
                    (
                        ann.bbox[0],
                        ann.bbox[1],
                        ann.bbox[2],
                        ann.bbox[3],
                        ann.category_id,
                    )
                    for ann in image_annotations[img.id]
                ]
            )
        else:
            Y.append([])

    X_tensor = tf.stack(X)
    Y_tensor = tf.stack(Y)
    return X_tensor, Y_tensor
