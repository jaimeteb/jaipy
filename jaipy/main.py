"""
Main module
"""


# from jaipy.dataset import get_images
# from jaipy.logger import logger
from jaipy.model import Model


def main():
    model = Model()
    model.train()
    model.test()
