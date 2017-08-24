from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

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

from keras.regularizers import l2

from keras import backend as K


def content_network():
    """Set up the content network.

    Ref: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    """
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1
