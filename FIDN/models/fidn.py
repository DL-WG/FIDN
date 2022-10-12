__author__ = "Bo Pang"
__copyright__ = "Copyright 2022, IRP Project"
__credits__ = ["Bo Pang"]
__license__ = "Apache 2.0"
__version__ = "1.0"
__email__ = "bo.pang21@imperial.ac.uk"

from tensorflow import keras
from keras.api._v2 import keras
from keras.layers import VersionAwareLayers

from FIDN.models.densenet import FIDNEncoder, conv_block

layers = VersionAwareLayers()

def build_fidn_model(inp):
    """
    Construction of the FIDN model,
    see report/bp221-final-report.pdf for details of the model structure
    
    Args:
        inp: Dimensions of the model input

    Returns:
        A keras instance of the FIDN model
    """
    border_size = int((inp.shape[2] - 128) / 2)
    other_inp = inp[:, border_size:-border_size, border_size:-border_size, ...]

    fire_block = [6, 12, 24, 6]
    other_block = [6, 12]

    # Build FIDN encoder for different layers
    dse_fire = FIDNEncoder(
        blocks=fire_block,
        input_shape=(*inp.shape[-3:-1], 3),
        name='dse_fire'
    )
    dse_biomass_above = FIDNEncoder(
        blocks=other_block,
        input_shape=(*other_inp.shape[-3:-1], 1),
        name='dse_biomass_above'
    )
    dse_biomass_under = FIDNEncoder(
        blocks=other_block,
        input_shape=(*other_inp.shape[-3:-1], 1),
        name='dse_biomass_under'
    )
    dse_slope = FIDNEncoder(
        blocks=other_block,
        input_shape=(*other_inp.shape[-3:-1], 1),
        name='dse_slope'
    )
    dse_tree_grass_bare_snow_water = FIDNEncoder(
        blocks=other_block,
        input_shape=(*other_inp.shape[-3:-1], 5),
        name='dse_tree_grass_bare_snow_water'
    )
    dse_wind = FIDNEncoder(
        blocks=other_block,
        input_shape=(*other_inp.shape[-3:-1], 2),
        name='dse_wind'
    )
    dse_precipitation = FIDNEncoder(
        blocks=other_block,
        input_shape=(*other_inp.shape[-3:-1], 1),
        name='dse_precipitation'
    )

    feature_fire = dse_fire(inp[..., 0:3])
    feature_biomass_above = dse_biomass_above(other_inp[..., 3:4])
    feature_biomass_under = dse_biomass_under(other_inp[..., 4:5])
    feature_slope = dse_slope(other_inp[..., 5:6])
    feature_tree_grass_bare_snow_water = dse_tree_grass_bare_snow_water(
        other_inp[..., 6:11])
    feature_wind = dse_wind(other_inp[..., 11:13])
    feature_precipitation = dse_precipitation(other_inp[..., 13])

    # Concat all of the feature maps
    x = layers.Concatenate(
        axis=-1)([feature_fire, feature_biomass_above, feature_biomass_under,
                  feature_slope, feature_tree_grass_bare_snow_water,
                  feature_wind, feature_precipitation])

    # Construct FIDN Decoder
    x = layers.Conv2DTranspose(256, kernel_size=2, strides=(
        2, 2), name='decoder_1_conv2dtranspose')(x)
    x = conv_block(x, 32, name='decoder_1_conv_block')
    x = layers.Conv2DTranspose(128, kernel_size=2, strides=(
        2, 2), name='decoder_2_conv2dtranspose')(x)
    x = conv_block(x, 32, name='decoder_2_conv_block')
    x = layers.Conv2DTranspose(64, kernel_size=2, strides=(
        2, 2), name='decoder_3_conv2dtranspose')(x)
    x = conv_block(x, 32, name='decoder_3_conv_block')
    x = layers.Conv2DTranspose(32, kernel_size=2, strides=(
        2, 2), name='decoder_4_conv2dtranspose')(x)
    x = conv_block(x, 32, name='decoder_4_conv_block')
    x = layers.Conv2DTranspose(16, kernel_size=2, strides=(
        2, 2), name='decoder_5_conv2dtranspose')(x)
    x = conv_block(x, 32, name='decoder_5_conv_block')
    x = layers.Conv2DTranspose(4, kernel_size=2, strides=(
        2, 2), name='decoder_6_conv2dtranspose')(x)
    x = conv_block(x, 32, name='decoder_6_conv_block')
    x = layers.Conv2D(
        filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same"
    )(x)

    model = keras.models.Model(inp, x)
    return model
