"""
General settings
"""

import typing as t

NUM_CLASSES: int = 5
INPUT_SIZE: int = 224
CHANNELS: int = 3

CLASSES: t.List[str] = ["person", "car", "stop sign", "traffic light", "bicycle"]
EXPORT_DIR: str = "./.fiftyone/coco-2017"

NUM_BOXES: int = 5
GRID: int = 7

SEED: int = 42

BATCH_SIZE: int = 64
EPOCHS: int = 5

PREDICTION_THRESHOLD: float = 0.01

TRAIN_CUTOFF_START: float = 0.000
TRAIN_CUTOFF_END: float = 0.020

VAL_CUTOFF_START: float = 0.020
VAL_CUTOFF_END: float = 0.025

TEST_CUTOFF_START: float = 0.99
TEST_CUTOFF_END: float = 1.00

TEST_BATCH_SIZE: int = 8
