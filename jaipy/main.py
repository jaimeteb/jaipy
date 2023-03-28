"""
Main module
"""


from jaipy.dataset import get_images
from jaipy.logger import logger
from jaipy.model import Model


def main():
    X, Y = get_images(1_000)

    model = Model()
    model.train(X, Y)

    pred = model.predict(X)
    logger.info(pred)


# def generate_annotations():
#     generate_image_annotations()
