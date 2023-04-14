"""
Test loss function.
"""

from jaipy import loss
from tests.testres import mock_data


def test_loss():
    y_true = mock_data.get_mock_tensor_one_box_one_class(add_dimension=True)
    y_pred = mock_data.get_mock_tensor_two_boxes_one_class(add_dimension=True)

    yolo_like_loss = loss.YOLOLikeLoss()
    loss_value = yolo_like_loss(y_true, y_pred)
    assert loss_value > 0.0
