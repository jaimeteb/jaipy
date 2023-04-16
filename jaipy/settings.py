"""
General settings
"""

import typing as t

from pydantic import BaseSettings


class Settings(BaseSettings):
    num_classes: int = 5
    input_size: int = 224
    channels: int = 3

    classes: t.List[str] = ["person", "car", "stop sign", "traffic light", "bicycle"]
    export_dir: str = "./.fiftyone/coco-2017"

    num_boxes: int = 5
    grid: int = 7

    seed: int = 42

    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.000_1

    prediction_threshold: float = 0.01

    train_cutoff_start: float = 0.00
    train_cutoff_end: float = 0.20

    val_cutoff_start: float = 0.20
    val_cutoff_end: float = 0.25

    test_cutoff_start: float = 0.99
    test_cutoff_end: float = 1.00

    test_batch_size: int = 16


settings = Settings()
