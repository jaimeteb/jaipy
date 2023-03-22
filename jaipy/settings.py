"""
General settings
"""

import typing as t

NUM_CLASS: int = 5
INPUT_SIZE: int = 224
CHANNELS: int = 3

CLASSES: t.List[str] = ["person", "car", "stop sign", "traffic light", "bicycle"]
EXPORT_DIR: str = "./.fiftyone/coco-2017"
