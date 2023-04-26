"""
Main module
"""

from jaipy.dataset import DataGenerator, generate_dataset_files
from jaipy.model import Model
from jaipy.settings import settings


def train():
    model = Model()
    model.train(
        train_data=DataGenerator(
            batch_size=settings.batch_size,
            cutoff_start=settings.train_cutoff_start,
            cutoff_end=settings.train_cutoff_end,
        ),
        val_data=DataGenerator(
            batch_size=settings.batch_size,
            cutoff_start=settings.val_cutoff_start,
            cutoff_end=settings.val_cutoff_end,
            shuffle=False,
        ),
    )


def train_test():
    model = Model()
    model.train(
        train_data=DataGenerator(
            batch_size=settings.batch_size,
            cutoff_start=settings.train_cutoff_start,
            cutoff_end=settings.train_cutoff_end,
        ),
        val_data=DataGenerator(
            batch_size=settings.batch_size,
            cutoff_start=settings.val_cutoff_start,
            cutoff_end=settings.val_cutoff_end,
            shuffle=False,
        ),
    )
    model.test(
        test_data=DataGenerator(
            batch_size=settings.test_batch_size,
            cutoff_start=settings.test_cutoff_start,
            cutoff_end=settings.test_cutoff_end,
            shuffle=False,
        )
    )


def test():
    if settings.weights_file is not None:
        model = Model()
        model.load_weights(settings.weights_file)
        model.test(
            test_data=DataGenerator(
                batch_size=settings.test_batch_size,
                cutoff_start=settings.test_cutoff_start,
                cutoff_end=settings.test_cutoff_end,
                shuffle=False,
            )
        )


def predict():
    if settings.weights_file is not None:
        model = Model()
        model.load_weights(settings.weights_file)

        test_data = DataGenerator(
            batch_size=settings.test_batch_size,
            cutoff_start=settings.test_cutoff_start,
            cutoff_end=settings.test_cutoff_end,
            shuffle=False,
        )
        X, _ = test_data[0]
        model.predict(X, nms=True)


def train_test_mock():
    model = Model(batch_size=1)

    dg = DataGenerator(test=True)
    model.train(train_data=dg, val_data=dg, checkpoints=False)
    model.test(test_data=dg, test_batch_size=1)


def generate_dataset():
    generate_dataset_files()
