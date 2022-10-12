__author__ = "Bo Pang"
__copyright__ = "Copyright 2022, IRP Project"
__credits__ = ["Bo Pang"]
__license__ = "Apache 2.0"
__version__ = "1.0"
__email__ = "bo.pang21@imperial.ac.uk"

"""
This file stores the various loss functions associated with FIDN
"""

import tensorflow as tf
from tensorflow import keras
from keras.api._v2 import keras


def scale2range(x, range):
    """
    Scale x into a range, both expected to be floats (Numpy version)
    
    Args:
      x (np.ndarray): input array.
      range (list): list Range of data to be scaled to

    Returns:
      Output tensor for the block.
    """
    if x.max() == x.min():
        return x
    return (x - x.min()) * (max(range) - min(range)) / (
            x.max() - x.min()) + min(range)


def scale2range_tf(x, range):
    """
    Scale x into a range, both expected to be floats (Tensorflow version)
    
    Args:
      x (tf.Tensor): input tenspr.
      range (list):  Range of data to be scaled to

    Returns:
      Output tensor for the block.
    """
    result = tf.add(
        tf.math.divide(
            tf.multiply(
                tf.subtract(x, tf.reduce_min(
                    x, axis=[1, 2, 3], keepdims=True)),
                max(range) - min(range)
            ),
            tf.subtract(
                tf.reduce_max(x, axis=[1, 2, 3], keepdims=True),
                tf.reduce_min(x, axis=[1, 2, 3], keepdims=True)
            )
        ),
        min(range)
    )
    result_not_nan = tf.dtypes.cast(tf.math.logical_not(
        tf.math.is_nan(result)), dtype=tf.float32)
    return tf.math.multiply_no_nan(result, result_not_nan)


def custom_mean_squared_error(y_true, y_pred):
    """
    Calculate the mean square error after scaling each 
    pixel of the image to the interval 0 to 1

    `loss = mean(square(y_true - y_pred))`

    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared error values of batch.
    """
    # y_pred = scale2range(y_pred, [y_true.min(), y_true.max()])
    y_pred = scale2range_tf(y_pred, [0, 1])
    y_true = scale2range_tf(y_true, [0, 1])
    return tf.reduce_mean(keras.losses.mean_squared_error(y_true, y_pred))


def ssim_metrics(y_true, y_pred):
    """
    Calculate the structural similarity after scaling each 
    pixel of the image to the interval 0 to 1

    This function is based on the standard SSIM implementation from:
    Wang, Z., Bovik, A. C., Sheikh, H. R., \& Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing.

    Args:
        y_true: Ground truth values. 4-D Tensor of shape [batch, height, width, channels] with only Positive Pixel Values.
        y_pred: The predicted values. 4-D Tensor of shape [batch, height, width, channels] with only Positive Pixel Values.

    Returns:
        Mean SSIM values of batch.
    """
    y_pred = scale2range_tf(y_pred, [0, 1])
    y_true = scale2range_tf(y_true, [0, 1])
    # max_val, _ = _get_max_min_val(y_true, y_pred)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def psnr_metrics(y_true, y_pred):
    """
    Calculate the Peak signal-to-noise ratio after scaling each 
    pixel of the image to the interval 0 to 1

    The last three dimensions of input are expected to be [height, width, depth].


    Args:
        y_true: Ground truth values. First set of images.
        y_pred: The predicted values. Second set of images.

    Returns:
        Mean PSNR values of batch.
    """
    y_pred = scale2range_tf(y_pred, [0, 1])
    y_true = scale2range_tf(y_true, [0, 1])
    # max_val, _ = _get_max_min_val(y_true, y_pred)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))


def relative_root_mean_squared_error(y_true, y_pred):
    """
    Calculate the relative root mean square error.

    `loss = sqrt(sum(square(y_true - y_pred)) / sum(square(y_true)))`

    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
        Relative root mean squared error values of batch.
    """
    # y_pred = scale2range_tf(y_pred, [0, 1])
    # y_true = scale2range_tf(y_true, [0, 1])
    num = tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2, 3])
    den = tf.reduce_sum(tf.square(y_pred), axis=[1, 2, 3])
    squared_error = num / den
    rrmse_loss = tf.sqrt(squared_error)
    return tf.reduce_mean(rrmse_loss)
