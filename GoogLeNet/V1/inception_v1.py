from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras.initializers import Constant


def inception_module(inpt,
                     filters_1,
                     filters_3_reduce,
                     filters_3,
                     filters_5_reduce,
                     filters_5,
                     filters_pool):

    conv_1 = Conv2D(filters=filters_1,
                    kernel_size=(1, 1),
                    padding='same',
                    activation='relu',
                    bias_initializer=Constant(0.2),
                    kernel_regularizer=l2(2e-4))(inpt)

    conv_3_reduce = Conv2D(filters=filters_3_reduce,
                           kernel_size=(1, 1),
                           padding='same',
                           activation='relu',
                           bias_initializer=Constant(0.2),
                           kernel_regularizer=l2(2e-4))(inpt)

    conv_3 = Conv2D(filters=filters_3,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='relu',
                    bias_initializer=Constant(0.2),
                    kernel_regularizer=l2(2e-4))(conv_3_reduce)

    conv_5_reduce = Conv2D(filters=filters_5_reduce,
                           kernel_size=(1, 1),
                           padding='same',
                           activation='relu',
                           bias_initializer=Constant(0.2),
                           kernel_regularizer=l2(2e-4))(inpt)

    conv_5 = Conv2D(filters=filters_5,
                    kernel_size=(5, 5),
                    padding='same',
                    activation='relu',
                    bias_initializer=Constant(0.2),
                    kernel_regularizer=l2(2e-4))(conv_5_reduce)

    maxpool = MaxPooling2D(pool_size=(3, 3),
                           strides=(1, 1),
                           padding='same')(inpt)

    maxpool_proj = Conv2D(filters=filters_pool,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='same',
                          activation='relu',
                          bias_initializer=Constant(0.2),
                          kernel_regularizer=l2(2e-4))(maxpool)

    output = concatenate([conv_1, conv_3, conv_5, maxpool_proj], axis=1)  # channel first

    return output
