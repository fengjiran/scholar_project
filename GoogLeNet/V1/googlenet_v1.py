"""GoogLeNet V1 for keras.

# Reference
- Going Deeper with Convolutions
"""
from keras.initializers import Constant
from keras.layers import (AveragePooling2D, Conv2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model
from keras.regularizers import l2

from inception_v1 import inception_module


def GoogLeNet_v1(input_shape=(3, 224, 224), classes=1000):
    img_input = Input(shape=input_shape)

    x = ZeroPadding2D(padding=(3, 3))(img_input)
    x = Conv2D(filters=64,
               kernel_size=(7, 7),
               strides=(2, 2),
               bias_initializer=Constant(0.2),
               activation='relu',
               kernel_regularizer=l2(2e-4))(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=64,
               kernel_size=(1, 1),
               padding='same',
               bias_initializer=Constant(0.2),
               activation='relu',
               kernel_regularizer=l2(2e-4))(x)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=192,
               kernel_size=(3, 3),
               bias_initializer=Constant(0.2),
               activation='relu',
               kernel_regularizer=l2(2e-4))(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = ZeroPadding2D(padding=(1, 1))(x)

    # Inception 3a
    x = inception_module(inpt=x,
                         filters_1=64,
                         filters_3_reduce=96,
                         filters_3=128,
                         filters_5_reduce=16,
                         filters_5=32,
                         filters_pool=32)

    x = ZeroPadding2D(padding=(1, 1))(x)

    # Inception 3b
    x = inception_module(inpt=x,
                         filters_1=128,
                         filters_3_reduce=128,
                         filters_3=192,
                         filters_5_reduce=32,
                         filters_5=96,
                         filters_pool=64)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # x = ZeroPadding2D(padding=(1, 1))(x)

    # Inception 4a
    x = inception_module(inpt=x,
                         filters_1=192,
                         filters_3_reduce=96,
                         filters_3=208,
                         filters_5_reduce=16,
                         filters_5=48,
                         filters_pool=64)

    branch_1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
    branch_1 = Conv2D(filters=128,
                      kernel_size=(1, 1),
                      activation='relu',
                      bias_initializer=Constant(0.2),
                      kernel_regularizer=l2(2e-4))(branch_1)

    branch_1 = Flatten()(branch_1)
    branch_1 = Dense(1024,
                     activation='relu',
                     bias_initializer=Constant(0.2),
                     kernel_regularizer=l2(2e-4))(branch_1)
    branch_1 = Dropout(0.7)(branch_1)
    branch_1 = Dense(classes,
                     activation='softmax',
                     kernel_regularizer=l2(2e-4),
                     name='branch_1')(branch_1)  # 0.3

    # x = ZeroPadding2D(padding=(1, 1))(x)

    # Inception 4b
    x = inception_module(inpt=x,
                         filters_1=160,
                         filters_3_reduce=112,
                         filters_3=224,
                         filters_5_reduce=24,
                         filters_5=64,
                         filters_pool=64)

    # x = ZeroPadding2D(padding=(1, 1))(x)

    # Inception 4c
    x = inception_module(inpt=x,
                         filters_1=128,
                         filters_3_reduce=128,
                         filters_3=256,
                         filters_5_reduce=24,
                         filters_5=64,
                         filters_pool=64)

    # x = ZeroPadding2D(padding=(1, 1))(x)

    # Inception 4d
    x = inception_module(inpt=x,
                         filters_1=112,
                         filters_3_reduce=144,
                         filters_3=288,
                         filters_5_reduce=32,
                         filters_5=64,
                         filters_pool=64)

    branch_2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
    branch_2 = Conv2D(filters=128,
                      kernel_size=(1, 1),
                      activation='relu',
                      bias_initializer=Constant(0.2),
                      kernel_regularizer=l2(2e-4))(branch_2)

    branch_2 = Flatten()(branch_2)
    branch_2 = Dense(1024,
                     activation='relu',
                     bias_initializer=Constant(0.2),
                     kernel_regularizer=l2(2e-4))(branch_2)
    branch_2 = Dropout(0.7)(branch_2)
    branch_2 = Dense(classes,
                     activation='softmax',
                     kernel_regularizer=l2(2e-4),
                     name='branch_2')(branch_2)  # 0.3

    # x = ZeroPadding2D(padding=(1, 1))(x)

    # Inception 4e
    x = inception_module(inpt=x,
                         filters_1=256,
                         filters_3_reduce=160,
                         filters_3=320,
                         filters_5_reduce=32,
                         filters_5=128,
                         filters_pool=128)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # x = ZeroPadding2D(padding=(1, 1))(x)

    # Inception 5a
    x = inception_module(inpt=x,
                         filters_1=256,
                         filters_3_reduce=160,
                         filters_3=320,
                         filters_5_reduce=32,
                         filters_5=128,
                         filters_pool=128)

    # x = ZeroPadding2D(padding=(1, 1))(x)
    # Inception 5b
    x = inception_module(inpt=x,
                         filters_1=384,
                         filters_3_reduce=192,
                         filters_3=384,
                         filters_5_reduce=48,
                         filters_5=128,
                         filters_pool=128)

    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    branch_3 = Dense(classes,
                     activation='softmax',
                     kernel_regularizer=l2(2e-4),
                     name='branch_3')(x)  # 1

    model = Model(inputs=img_input, outputs=[branch_1, branch_2, branch_3])

    return model


if __name__ == '__main__':
    model = GoogLeNet_v1()
    print model.summary()
