"""
Logging module
"""
# pylint: disable=redefined-outer-name

import logging

import colorlog


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    stream_handler = colorlog.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
    )

    logger.addHandler(stream_handler)

    return logger


logger = get_logger(__name__)
