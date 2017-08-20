from __future__ import division

import os
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras import backend as K

import numpy as np


class Encoder(object):
    """The encoder is derived from AlexNet.

    Using the first five convolutional layers
    and the following pooling layer to compute
    an abstract 6*6*256 dimensional feature representation.
    """

    def __init__(self, activation='relu'):

        self.activation = activation

        self.model = Sequential()
        self.model.add(Conv2D(filters=96,
                              kernel_size=(11, 11),
                              strides=(4, 4),
                              input_shape=(3, 227, 227),
                              activation='relu',
                              name='conv1'))

        self.model.add(MaxPooling2D(pool_size=(3, 3),
                                    strides=(2, 2)))

        self.model.add(ZeroPadding2D(padding=(2, 2)))

        self.model.add(Conv2D(filters=256,
                              kernel_size=(5, 5),
                              activation='relu',
                              name='conv2'))

        self.model.add(MaxPooling2D(pool_size=(3, 3),
                                    strides=(2, 2)))

        self.model.add(ZeroPadding2D(padding=(1, 1)))

        self.model.add(Conv2D(filters=384,
                              kernel_size=(3, 3),
                              activation='relu',
                              name='conv3'))

        self.model.add(ZeroPadding2D(padding=(1, 1)))

        self.model.add(Conv2D(filters=384,
                              kernel_size=(3, 3),
                              activation='relu',
                              name='conv4'))

        self.model.add(ZeroPadding2D(padding=(1, 1)))

        self.model.add(Conv2D(filters=256,
                              kernel_size=(3, 3),
                              activation='relu',
                              name='conv5'))

        self.model.add(MaxPooling2D(pool_size=(3, 3),
                                    strides=(2, 2)))

        print self.model.summary()


class Channel_wise_fc_layer(object):
    """Channel-wise fully-connected layer."""

    def __init__(self):
        pass
