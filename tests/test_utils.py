"""
Unit tests for the utils module.
"""

from jaipy import utils
from tests.testres import mock_data


def test_draw_bounding_boxes():
    tensor = mock_data.get_mock_tensor_one_box_one_class()
    image = mock_data.get_empty_image()

    image_true = utils.draw_bounding_boxes(image, tensor)
    image_pred = utils.draw_bounding_boxes(image, tensor, pred=True)

    assert image_true == image_pred
