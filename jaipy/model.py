"""
Neural network module definition
"""
# pyright: reportMissingImports=false
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Input,
    LeakyReLU,
    MaxPool2D,
    ZeroPadding2D,
)
from tensorflow.keras.regularizers import l2

STRIDES = np.array([16, 32])
ANCHORS = (
    np.array(
        [
            [[10, 14], [23, 27], [37, 58]],
            [[81, 82], [135, 169], [344, 319]],
        ]
    ).T
    / STRIDES
).T


def Convolutional(layer, filters_shape, activate=True, bn=True):
    strides = 1
    padding = "same"

    layer = Conv2D(
        filters=filters_shape[-1],
        kernel_size=filters_shape[0],
        strides=strides,
        padding=padding,
        use_bias=not bn,
        kernel_regularizer=l2(0.000_5),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0),
    )(layer)
    layer = BatchNormalization()(layer) if bn else layer
    layer = LeakyReLU(alpha=0.1)(layer) if activate else layer

    return layer


def UpSample(layer):
    return tf.image.resize(
        layer,
        (
            layer.shape[1] * 2,
            layer.shape[2] * 2,
        ),
        method="nearest",
    )


def YOLOv3_tiny(layer, NUM_CLASS):
    # darknet19_tiny(layer):
    layer = Convolutional(layer, (3, 3, 3, 16))
    layer = MaxPool2D(2, 2, "same")(layer)
    layer = Convolutional(layer, (3, 3, 16, 32))
    layer = MaxPool2D(2, 2, "same")(layer)
    layer = Convolutional(layer, (3, 3, 32, 64))
    layer = MaxPool2D(2, 2, "same")(layer)
    layer = Convolutional(layer, (3, 3, 64, 128))
    layer = MaxPool2D(2, 2, "same")(layer)
    layer = Convolutional(layer, (3, 3, 128, 256))
    layer_scale = layer
    layer = MaxPool2D(2, 2, "same")(layer)
    layer = Convolutional(layer, (3, 3, 256, 512))
    layer = MaxPool2D(2, 1, "same")(layer)
    layer = Convolutional(layer, (3, 3, 512, 1024))

    layer = Convolutional(layer, (1, 1, 1024, 256))
    layer_lobj_branch = Convolutional(layer, (3, 3, 256, 512))

    # layer_lbbox is used to predict large-sized objects , Shape = [None, 26, 26, 255]
    layer_lbbox = Convolutional(
        layer_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False
    )

    layer = Convolutional(layer, (1, 1, 256, 128))
    # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
    # upsampling process does not need to learn, thereby reducing the network parameter
    layer = UpSample(layer)

    layer = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        [layer, layer_scale], axis=-1
    )
    layer_mobj_branch = Convolutional(layer, (3, 3, 128, 256))
    # layer_mbbox is used to predict medium size objects, shape = [None, 13, 13, 255]
    layer_mbbox = Convolutional(
        layer_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False
    )

    return [layer_mbbox, layer_lbbox]


def decode(conv_output, NUM_CLASS, i=0):
    # where i = 0, 1 or 2 to correspond to the three grid scales
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(
        conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS)
    )

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]  # offset of center position
    conv_raw_dwdh = conv_output[
        :, :, :, :, 2:4
    ]  # Prediction box length and width offset
    conv_raw_conf = conv_output[:, :, :, :, 4:5]  # confidence of the prediction box
    conv_raw_prob = conv_output[
        :, :, :, :, 5:
    ]  # category probability of the prediction box

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52
    y = tf.range(output_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size])
    x = tf.range(output_size, dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [output_size, 1])

    xy_grid = (
        tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
            [x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1
        )
    )
    xy_grid = tf.tile(
        xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1]
    )
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # Calculate the length and width of the prediction box:
    pred_wh = (  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        tf.exp(conv_raw_dwdh) * ANCHORS[i]
    ) * STRIDES[i]

    pred_xywh = (
        tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
            [pred_xy, pred_wh], axis=-1
        )
    )
    pred_conf = tf.sigmoid(
        conv_raw_conf
    )  # object box calculates the predicted confidence
    pred_prob = tf.sigmoid(
        conv_raw_prob
    )  # calculating the predicted probability category box object

    # calculating the predicted probability category box object
    return tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        [pred_xywh, pred_conf, pred_prob], axis=-1
    )


def Create_Yolov3(input_size=416, channels=3, training=False):
    NUM_CLASS = 5
    input_layer = Input([input_size, input_size, channels])

    conv_tensors = YOLOv3_tiny(input_layer, NUM_CLASS)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, NUM_CLASS, i)
        if training:
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    YoloV3 = tf.keras.Model(input_layer, output_tensors)
    return YoloV3


net = Create_Yolov3(training=True)
