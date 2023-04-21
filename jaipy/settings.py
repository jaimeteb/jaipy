"""
General settings
"""

import typing as t

from pydantic import BaseSettings


class Settings(BaseSettings):
    num_classes: int = 5
    input_size: int = 448
    channels: int = 3

    classes: t.List[str] = ["person", "car", "stop sign", "traffic light", "bicycle"]
    export_dir: str = "./.fiftyone/coco-2017"

    num_boxes: int = 5
    grid: int = 7

    seed: int = 42

    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 0.001

    prediction_threshold: float = 0.000_1

    train_cutoff_start: float = 0.00
    train_cutoff_end: float = 0.01

    val_cutoff_start: float = 0.01
    val_cutoff_end: float = 0.02

    test_cutoff_start: float = 0.998
    test_cutoff_end: float = 0.999

    test_batch_size: int = 16


settings = Settings()
