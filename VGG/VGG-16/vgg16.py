"""VGG16 model for Keras.

# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import division

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.regularizers import l2
# from keras.layers.core import Activation
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
# from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D


def VGG16(input_shape=(3, 224, 224), classes=1000):
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(5e-4), name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4), name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4), name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4), name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4), name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4), name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4), name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4), name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4), name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4), name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4), name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4), name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4), name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Block6
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    output = Dense(classes, activation='softmax', name='predictions')(x)

    # Create the model.
    model = Model(inputs=img_input, outputs=output, name='vgg16')

    return model


if __name__ == '__main__':
    model = VGG16()
    print model.summary()
