"""
Main module
"""


from jaipy.dataset import get_images  # , generate_image_annotations
from jaipy.logger import logger
from jaipy.model import Model


def main():
    X = get_images(4)

    model = Model()
    pred = model.predict(X)
    logger.info(pred)


# def generate_annotations():
#     generate_image_annotations()
