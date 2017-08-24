from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from keras import layers
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Reshape
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ZeroPadding2D

from keras.regularizers import l2

from keras import backend as K


def identity_block(input_tensor, kernel_size, filters):
    """Constuct the identity block(pre-activation).

    The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = BatchNormalization(axis=bn_axis)(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters1,
               kernel_size=(1, 1),
               kernel_regularizer=l2(1e-4))(x)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters2,
               kernel_size=kernel_size,
               kernel_regularizer=l2(1e-4),
               padding='same')(x)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters3,
               kernel_size=(1, 1),
               kernel_regularizer=l2(1e-4))(x)

    x = layers.add([x, input_tensor])
    # x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    """Construct a block that has a conv layer at shortcut(pre-activation).

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path

    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = BatchNormalization(axis=bn_axis)(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters1,
               strides=strides,
               kernel_size=(1, 1),
               kernel_regularizer=l2(1e-4))(x)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters2,
               kernel_size=kernel_size,
               kernel_regularizer=l2(1e-4),
               padding='same')(x)

    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters3,
               kernel_size=(1, 1),
               kernel_regularizer=l2(1e-4))(x)

    shortcut = BatchNormalization(axis=bn_axis)(input_tensor)
    shortcut = Activation('relu')(shortcut)
    shortcut = Conv2D(filters=filters3,
                      strides=strides,
                      kernel_size=(1, 1),
                      kernel_regularizer=l2(1e-4))(shortcut)

    x = layers.add([x, shortcut])

    return x


def content_network():
    """Set up the content network.

    Ref: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    """
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1

    img_input = Input(shape=(3, 224, 224))

    x = Conv2D(filters=64,
               kernel_size=(7, 7),
               strides=(2, 2),
               padding='same',
               activation='relu')(img_input)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # (N, 64, 56, 56)

    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])    # (N, 256, 56, 56)

    model = Model(img_input, x)

    return model


if __name__ == '__main__':
    model = content_network()
    print(model.summary())
