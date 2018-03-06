# -*- coding: utf-8 -*-
"""ResNet50 model for Keras.

# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D

from keras import backend as K

from resnet_utils import identity_block
from resnet_utils import conv_block


def ResNet50(input_shape=(3, 224, 224), classes=1000):
    img_input = Input(shape=input_shape)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # (N, 64, 56, 56)

    # stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')  # (N, 256, 56, 56)

    # stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')  # (N, 512, 28, 28)

    # stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')  # (N, 1024, 14, 14)

    # stage 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')  # (N, 2048, 7, 7)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)  # (N, 2048, 1, 1)
    x = Flatten()(x)  # (N, 2048)
    x = Dense(classes, activation='softmax', name='fc1000')(x)  # (N, 1000)
    # x = GlobalMaxPooling2D()(x)
    # x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')
    return model


if __name__ == '__main__':
    # inpt = Input(shape=(256, 224, 224))
    # output = identity_block(inpt, (3, 3), (64, 64, 256), 1, 'a')
    # output = conv_block(inpt, (3, 3), (64, 64, 256), 1, 'a')

    model = ResNet50()
    print model.summary()
