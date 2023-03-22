"""
Main module
"""

from jaipy.dataset import get_images
from jaipy.model import Model


def main():
    X = get_images(4)

    model = Model()
    pred = model.predict(X)
    print(pred)


if __name__ == "__main__":
    main()
