"""
Test dataset
"""

import tensorflow as tf

from jaipy import dataset
from tests.testres import mock_data


def test_convert_annotations_to_yolo_like_tensor():
    anns, img, category_indices = mock_data.get_mock_annotations_image_category_indces()
    Y_expected = mock_data.get_mock_tensor_one_box_one_class_v2()
    Y_got = dataset.convert_annotations_to_yolo_like_tensor(anns, img, category_indices)
    assert tf.reduce_all(tf.equal(Y_expected, Y_got))
