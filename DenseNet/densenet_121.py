from keras.models import Model
from keras.layers import concatenate
from keras.layers import Input
from keras.layers import Dense
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


def conv_block(x, stage, branch, nb_filters, bottleneck=True, dropout_rate=None, weight_decay=1e-4):
    """Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout.

    # Arguments
        x: input tensor
        stage: index for dense block
        branch: layer index within each dense block
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    """
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # 1x1 Convolution (Bottleneck layer)
    if bottleneck:
        inter_channel = nb_filters * 4

        x = BatchNormalization(axis=bn_axis,
                               gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay),
                               name=conv_name_base + '_x1_bn')(x)

        x = Activation('relu', name=relu_name_base + '_x1')(x)
        x = Conv2D(filters=inter_channel,
                   kernel_size=(1, 1),
                   kernel_initializer='he_uniform',
                   kernel_regularizer=l2(weight_decay),
                   name=conv_name_base + '_x1')(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    # 3x3 convolutional
    x = BatchNormalization(axis=bn_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay),
                           name=conv_name_base + '_x2_bn')(x)

    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = Conv2D(filters=nb_filters,
               kernel_size=(3, 3),
               padding='same',
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay),
               name=conv_name_base + '_x2')(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filters, compression=0.5, dropout_rate=None, weight_decay=1e-4):
    """Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout.

    # Arguments
        x: input tensor
        stage: index for dense block
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    """
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = BatchNormalization(axis=bn_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay),
                           name=conv_name_base + '_bn')(x)

    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(filters=int(nb_filters * compression),
               kernel_size=(1, 1),
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay),
               name=conv_name_base)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D(pool_size=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filters, growth_rate, dropout_rate=None, weight_decay=1e-4):
    """Build a dense_block where the output of each conv_block is fed to subsequent ones.

    # Arguments
        x: input tensor
        stage: index for dense block
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    """
    list_feat = [x]

    if K.image_data_format() == 'channels_first':
        concat_axis = 1
    else:
        concat_axis = -1

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(x=x,
                       stage=stage,
                       branch=branch,
                       nb_filters=growth_rate,
                       dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
        list_feat.append(x)
        x = concatenate(list_feat,
                        axis=concat_axis,
                        name='concat_' + str(stage) + '_' + str(branch))

        nb_filters += growth_rate

    return x, nb_filters


def DenseNet_121(nb_dense_block=4, growth_rate=32, nb_filters=64, compression=0.5,
                 dropout_rate=None, weight_decay=1e-4, classes=1000):
    """Instantiate the DenseNet 121 architecture.

    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        compression: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
        img_input = Input(shape=(224, 224, 3), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(3, 224, 224), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_layers = [6, 12, 24, 16]

    # Initial convolution
    x = ZeroPadding2D(padding=(3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(filters=nb_filters,
               kernel_size=(7, 7),
               strides=(2, 2),
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay),
               name='conv1')(x)

    x = BatchNormalization(axis=bn_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay),
                           name='conv1_bn')(x)

    x = Activation('relu', name='relu1')(x)

    x = ZeroPadding2D(padding=(1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D(pool_size=(3, 3),
                     strides=(2, 2),
                     name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filters = dense_block(x, stage, nb_layers[block_idx], nb_filters,
                                    growth_rate=growth_rate,
                                    dropout_rate=dropout_rate,
                                    weight_decay=weight_decay)

        # add transition
        x = transition_block(x, stage, nb_filters,
                             compression=compression,
                             dropout_rate=dropout_rate,
                             weight_decay=dropout_rate)

        nb_filters = int(nb_filters * compression)

    # The last denseblock does not have a transition
    final_stage = stage + 1
    x, nb_filters = dense_block(x, final_stage, nb_layers[-1], nb_filters,
                                growth_rate=growth_rate,
                                dropout_rate=dropout_rate,
                                weight_decay=weight_decay)

    x = BatchNormalization(axis=bn_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay),
                           name='conv' + str(final_stage) + '_blk_bn')(x)

    x = Activation('relu', name='conv' + str(final_stage) + '_blk')(x)
    x = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)

    x = Dense(classes, name='fc6')(x)
    x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='DenseNet_121')

    return model


if __name__ == '__main__':
    model = DenseNet_121()

    print model.summary()
