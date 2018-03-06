from keras.models import Model
from keras.layers import concatenate
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers import Dropout
from keras.regularizers import l2

from keras import backend as K


def conv_block(input_tensor, nb_filter, dropout_rate=None, weight_decay=1e-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout.

    :param input_tensor: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = BatchNormalization(axis=bn_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(input_tensor)

    x = Activation('relu')(x)
    x = Conv2D(filters=nb_filter,
               kernel_size=(3, 3),
               padding='same',
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(input_tensor, nb_filter, dropout_rate=None, weight_decay=1e-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D.

    :param input_tensor: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = BatchNormalization(axis=bn_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(input_tensor)

    x = Activation('relu')(x)
    x = Conv2D(filters=nb_filter,
               kernel_size=(1, 1),
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D(pool_size=(2, 2))(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4):
    """Build a denseblock where the output of each conv_block is fed to subsequent ones.

    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_block to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_block appended
    :rtype: keras model
    """
    list_feat = [x]

    if K.image_data_format() == 'channels_first':
        concat_axis = 1
    else:
        concat_axis = -1

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        list_feat.append(x)
        x = concatenate(list_feat, axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter


def DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate,
             nb_filter, dropout_rate=None, weight_decay=1e-4):
    """Build the DenseNet model(the densenet has three dense block).

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    model_input = Input(shape=img_dim)
    assert (depth - nb_dense_block - 1) % nb_dense_block == 0, "Depth must be 3N + 4"

    # layers in each dense block
    nb_layers = int((depth - nb_dense_block - 1) / nb_dense_block)

    # Initial convolution
    x = Conv2D(filters=nb_filter,
               kernel_size=(3, 3),
               padding='same',
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate,
                                   dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        # add transition
        x = transition(x, nb_filter, dropout_rate, weight_decay)

    # The last denseblock does not have a transition
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate,
                               dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = BatchNormalization(axis=bn_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    model = Model(model_input, x, name='DenseNet')

    return model


if __name__ == '__main__':
    model = DenseNet(nb_classes=10,
                     img_dim=(3, 32, 32),
                     depth=40,
                     nb_dense_block=3,
                     growth_rate=12,
                     nb_filter=16,
                     dropout_rate=0.2)

    print model.summary()
