__author__ = "Bo Pang"
__copyright__ = "Copyright 2022, IRP Project"
__credits__ = ["Bo Pang"]
__license__ = "Apache 2.0"
__version__ = "1.0"
__email__ = "bo.pang21@imperial.ac.uk"

"""
DenseNet models



Reference:
  - [Keras Applications](https://github.com/keras-team/keras-applications/blob/bc89834ed36935ab4a4994446e34ff81c0d8e1b7/keras_applications/densenet.py) 
  - [Densely Connected Convolutional Networks](
      https://arxiv.org/abs/1608.06993) (CVPR 2017)
"""

from keras import backend
from keras.engine import training
from keras.layers import VersionAwareLayers

layers = VersionAwareLayers()


def dense_block(x, blocks, name, channel_last=True):
    """
    A dense block.

    Args:
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
        channel_last: Indicates whether the last dimension of the image represents channel.

    Returns:
        Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' +
                                   str(i + 1), channel_last=channel_last)
    return x


def transition_block(x, reduction, name, channel_last=True, ):
    """
    A transition block.

    Args:
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
        channel_last: Indicates whether the last dimension of the image represents channel.

    Returns:
        output tensor for the block.
    """
    bn_axis = 3 if channel_last else 1
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(
        x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
        name=name + '_conv')(
        x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name, channel_last=True):
    """
    A building block for a dense block.

    Args:
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
        channel_last: Indicates whether the last dimension of the image represents channel.

    Returns:
        Output tensor for the block.
    """
    bn_axis = 3 if channel_last else 1
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
        x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(
        4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(
        x1)
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
        x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(
        growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
        x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def FIDNEncoder(
        blocks,
        input_shape=None,
        channel_last=True,
        name='FIDN'):
    """
    Instantiates the FIDN Encoder architecture.
    
    Args:
        blocks: numbers of building blocks for the four dense layers.
        input_shape: optional shape tuple.
        channel_last: Indicates whether the last dimension of the image represents channel.
        name: Name of the model, default is FIDN

    Returns:
        A keras.Model instance.
    """
    img_input = layers.Input(shape=input_shape)

    bn_axis = 3 if channel_last else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False,
                      name=f'{name}-conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=f'{name}-conv1/bn')(
        x)
    x = layers.Activation('relu', name=f'{name}-conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name=f'{name}-pool1')(x)

    for index, block in enumerate(blocks):
        x = dense_block(
            x, block, name=f'{name}-conv{index + 2}',
            channel_last=channel_last)
        x = transition_block(
            x, 0.5, name=f'{name}-pool{index + 2}', channel_last=channel_last)

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=f'{name}-bn')(x)
    x = layers.Activation('relu', name=f'{name}-relu')(x)

    inputs = img_input
    # Create model.
    model = training.Model(inputs, x, name=name)
    return model
