"""
Image dataset loading and parsing
"""

import os
import typing as t

import fiftyone as fo
import fiftyone.zoo as foz
from pydantic import BaseModel, parse_file_as  # pylint: disable=no-name-in-module

CLASSES = ["person", "car", "stop sign", "traffic light", "bicycle"]
EXPORT_DIR = "./.fiftyone/coco-2017"


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
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        splits=["train", "test", "validation"],
        label_types=["detections"],
        classes=CLASSES,
    )
    dataset.export(".fiftyone/coco-2017", fo.types.COCODetectionDataset)
    dataset.delete()


def get_dataset_labels() -> DatasetLabels:
    labels = parse_file_as(DatasetLabels, os.path.join(EXPORT_DIR, "labels.json"))
    return labels
