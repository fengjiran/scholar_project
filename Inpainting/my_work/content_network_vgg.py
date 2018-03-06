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

from imagenet_utils import _obtain_input_shape

WEIGHTS_PATH = './model/vgg19_weights_th_dim_ordering_th_kernels.h5'
WEIGHTS_PATH_NO_TOP = './model/vgg19_weights_th_dim_ordering_th_kernels_notop.h5'


def VGG19(include_top=False,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling='max',
          classes=1000):
    """Instantiate the VGG19 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.

        weights: one of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).

        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.

        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.

        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Create model.
    model = Model(img_input, x, name='vgg19')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = WEIGHTS_PATH
        else:
            weights_path = WEIGHTS_PATH_NO_TOP

    model.load_weights(weights_path)

    model.trainable = False

    # for layer in model.layers:
    #     layer.trainable = False

    return model


def content_network():
    """Set up the content network."""
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1

    encoder = VGG19(include_top=False, pooling='max')
    print('VGG 19 loaded!')

    content = Sequential()
    content.add(encoder)
    content.add(Dense(4000, activation='relu', kernel_regularizer=l2(5e-4)))
    content.add(Reshape((4000, 1, 1)))
    content.add(UpSampling2D(size=(2, 2)))
    content.add(Conv2DTranspose(filters=512,
                                kernel_size=(3, 3),
                                data_format=K.image_data_format(),
                                kernel_regularizer=l2(5e-4)))
    content.add(BatchNormalization(axis=bn_axis))
    content.add(Activation('relu'))

    content.add(UpSampling2D(size=(2, 2)))
    content.add(Conv2DTranspose(filters=256,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=K.image_data_format(),
                                kernel_regularizer=l2(5e-4)))
    content.add(BatchNormalization(axis=bn_axis))
    content.add(Activation('relu'))

    content.add(UpSampling2D(size=(2, 2)))
    content.add(Conv2DTranspose(filters=128,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=K.image_data_format(),
                                kernel_regularizer=l2(5e-4)))
    content.add(BatchNormalization(axis=bn_axis))
    content.add(Activation('relu'))

    content.add(UpSampling2D(size=(2, 2)))
    content.add(Conv2DTranspose(filters=64,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=K.image_data_format(),
                                kernel_regularizer=l2(5e-4)))
    content.add(BatchNormalization(axis=bn_axis))
    content.add(Activation('relu'))

    content.add(UpSampling2D(size=(2, 2)))
    content.add(Conv2DTranspose(filters=3,
                                kernel_size=(3, 3),
                                padding='same',
                                data_format=K.image_data_format(),
                                kernel_regularizer=l2(5e-4)))

    content.add(Activation('tanh'))
    return content


if __name__ == '__main__':
    # model = VGG19(include_top=False, pooling='max')
    # print(model.trainable)
    # print(model.summary())
    # print(model.layers[0].get_config())
    # print(model.layers[1].get_config())
    # print(model.layers[1].trainable)

    model = content_network()
    print(model.summary())

    a = np.random.rand(1, 3, 224, 224)
    b = model.predict(a)
    print(b.shape)
