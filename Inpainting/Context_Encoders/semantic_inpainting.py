from __future__ import division

from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU
from keras.layers.core import Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.regularizers import l2
from keras.optimizers import SGD
from keras import backend as K

import numpy as np


class Context_Encoder(object):
    """The context encoder for semantic inpainting."""

    def __init__(self,
                 encoder_input_shape=(3, 128, 128),
                 dis_input_shape=(3, 64, 64),
                 weight_decay_rate=0.00001):
        self.encoder_input_shape = encoder_input_shape
        self.dis_input_shape = dis_input_shape
        self.weight_decay_rate = weight_decay_rate

        self.encoder_input = None
        self.encoder_output = None

        self.dis_input = None
        self.dis_output = None

        self.recon = None
        self.encoder_decoder = None
        self.discriminator = None

        self.setup_model()

    def setup_model(self):
        """Method to set up model."""
        self.setup_encoder_decoder()
        self.setup_discriminator()

    def setup_encoder_decoder(self):
        """Method to set up encoder and decoder."""
        # Encoder
        self.encoder_input = Input(shape=self.encoder_input_shape)
        x = ZeroPadding2D(padding=(1, 1))(self.encoder_input)
        conv1 = Conv2D(filters=64,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       kernel_regularizer=l2(self.weight_decay_rate),
                       name='conv1')(x)
        bn1 = BatchNormalization(axis=1)(conv1)
        x = Activation(LeakyReLU(0.1))(bn1)

        x = ZeroPadding2D(padding=(1, 1))(x)
        conv2 = Conv2D(filters=64,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       kernel_regularizer=l2(self.weight_decay_rate),
                       name='conv2')(x)
        bn2 = BatchNormalization(axis=1)(conv2)
        x = Activation(LeakyReLU(0.1))(bn2)

        x = ZeroPadding2D(padding=(1, 1))(x)
        conv3 = Conv2D(filters=128,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       kernel_regularizer=l2(self.weight_decay_rate),
                       name='conv3')(x)
        bn3 = BatchNormalization(axis=1)(conv3)
        x = Activation(LeakyReLU(0.1))(bn3)

        x = ZeroPadding2D(padding=(1, 1))(x)
        conv4 = Conv2D(filters=256,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       kernel_regularizer=l2(self.weight_decay_rate),
                       name='conv4')(x)
        bn4 = BatchNormalization(axis=1)(conv4)
        x = Activation(LeakyReLU(0.1))(bn4)

        x = ZeroPadding2D(padding=(1, 1))(x)
        conv5 = Conv2D(filters=512,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       kernel_regularizer=l2(self.weight_decay_rate),
                       name='conv5')(x)
        bn5 = BatchNormalization(axis=1)(conv5)
        x = Activation(LeakyReLU(0.1))(bn5)

        conv6 = Conv2D(filters=4000,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       kernel_regularizer=l2(self.weight_decay_rate),
                       name='conv6')(x)
        bn6 = BatchNormalization(axis=1)(conv6)
        self.encoder_output = Activation(LeakyReLU(0.1))(bn6)

        x = UpSampling2D(size=(2, 2))(self.encoder_output)

        # Decoder
        deconv4 = Conv2DTranspose(filters=512,
                                  #   strides=(2, 2),
                                  kernel_size=(3, 3),
                                  data_format=K.image_data_format())(x)
        debn4 = BatchNormalization(axis=1)(deconv4)
        x = Activation('relu')(debn4)

        x = UpSampling2D(size=(2, 2))(x)
        deconv3 = Conv2DTranspose(filters=256,
                                  #   strides=(2, 2),
                                  kernel_size=(3, 3),
                                  padding='same',
                                  kernel_regularizer=l2(self.weight_decay_rate),
                                  data_format=K.image_data_format())(x)
        debn3 = BatchNormalization(axis=1)(deconv3)
        x = Activation('relu')(debn3)

        x = UpSampling2D(size=(2, 2))(x)
        deconv2 = Conv2DTranspose(filters=128,
                                  #   strides=(2, 2),
                                  kernel_size=(3, 3),
                                  padding='same',
                                  kernel_regularizer=l2(self.weight_decay_rate),
                                  data_format=K.image_data_format())(x)
        debn2 = BatchNormalization(axis=1)(deconv2)
        x = Activation('relu')(debn2)

        x = UpSampling2D(size=(2, 2))(x)
        deconv1 = Conv2DTranspose(filters=64,
                                  #   strides=(2, 2),
                                  kernel_size=(3, 3),
                                  padding='same',
                                  kernel_regularizer=l2(self.weight_decay_rate),
                                  data_format=K.image_data_format())(x)
        debn1 = BatchNormalization(axis=1)(deconv1)
        x = Activation('relu')(debn1)

        x = UpSampling2D(size=(2, 2))(x)
        self.recon = Conv2DTranspose(filters=3,
                                     #  strides=(2, 2),
                                     kernel_size=(3, 3),
                                     padding='same',
                                     kernel_regularizer=l2(self.weight_decay_rate),
                                     data_format=K.image_data_format())(x)

        self.encoder_decoder = Model(inputs=self.encoder_input,
                                     outputs=self.recon,
                                     name='encoder_decoder')

        print self.encoder_decoder.summary()
        # self.model = Sequential()
        # self.model.add(ZeroPadding2D(padding=(1, 1),
        #                              input_shape=(3, 128, 128)))
        # self.model.add(Conv2D(filters=64,
        #                       kernel_size=(4, 4),
        #                       strides=(2, 2),
        #                       activation=LeakyReLU(0.1),
        #                       name='conv1'))

        # self.model.add(BatchNormalization(axis=1))

        # self.model.add(ZeroPadding2D(padding=(1, 1)))
        # self.model.add(Conv2D(filters=64,
        #                       kernel_size=(4, 4),
        #                       strides=(2, 2),
        #                       activation=LeakyReLU(0.1),
        #                       name='conv2'))

        # self.model.add(BatchNormalization(axis=1))

        # self.model.add(ZeroPadding2D(padding=(1, 1)))
        # self.model.add(Conv2D(filters=128,
        #                       kernel_size=(4, 4),
        #                       strides=(2, 2),
        #                       activation=LeakyReLU(0.1),
        #                       name='conv3'))

        # self.model.add(BatchNormalization(axis=1))

        # self.model.add(ZeroPadding2D(padding=(1, 1)))
        # self.model.add(Conv2D(filters=256,
        #                       kernel_size=(4, 4),
        #                       strides=(2, 2),
        #                       activation=LeakyReLU(0.1),
        #                       name='conv4'))

        # self.model.add(BatchNormalization(axis=1))

        # self.model.add(ZeroPadding2D(padding=(1, 1)))
        # self.model.add(Conv2D(filters=512,
        #                       kernel_size=(4, 4),
        #                       strides=(2, 2),
        #                       activation=LeakyReLU(0.1),
        #                       name='conv5'))

        # self.model.add(BatchNormalization(axis=1))

        # self.model.add(Conv2D(filters=4000,
        #                       kernel_size=(4, 4),
        #                       strides=(2, 2),
        #                       activation=LeakyReLU(0.1),
        #                       name='conv6'))

        # self.model.add(BatchNormalization(axis=1))

    def setup_discriminator(self):
        """Method to set up discriminator."""
        self.dis_input = Input(shape=self.dis_input_shape)
        x = ZeroPadding2D(padding=(1, 1))(self.dis_input)
        conv1 = Conv2D(filters=64,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       kernel_regularizer=l2(self.weight_decay_rate),
                       name='dis_conv1')(x)
        # bn1 = BatchNormalization(axis=1)(conv1)
        # x = Activation(LeakyReLU(0.1))(bn1)
        x = Activation(LeakyReLU(0.1))(conv1)

        x = ZeroPadding2D(padding=(1, 1))(x)
        conv2 = Conv2D(filters=128,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       kernel_regularizer=l2(self.weight_decay_rate),
                       name='dis_conv2')(x)
        # bn2 = BatchNormalization(axis=1)(conv2)
        # x = Activation(LeakyReLU(0.1))(bn2)
        x = Activation(LeakyReLU(0.1))(conv2)

        x = ZeroPadding2D(padding=(1, 1))(x)
        conv3 = Conv2D(filters=256,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       kernel_regularizer=l2(self.weight_decay_rate),
                       name='dis_conv3')(x)
        # bn3 = BatchNormalization(axis=1)(conv3)
        # x = Activation(LeakyReLU(0.1))(bn3)
        x = Activation(LeakyReLU(0.1))(conv3)

        x = ZeroPadding2D(padding=(1, 1))(x)
        conv4 = Conv2D(filters=512,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       kernel_regularizer=l2(self.weight_decay_rate),
                       name='dis_conv4')(x)
        # bn4 = BatchNormalization(axis=1)(conv4)
        # x = Activation(LeakyReLU(0.1))(bn4)
        x = Activation(LeakyReLU(0.1))(conv4)

        x = GlobalAveragePooling2D()(x)
        self.dis_output = Dense(1,
                                activation='sigmoid',
                                kernel_regularizer=l2(self.weight_decay_rate))(x)

        self.discriminator = Model(inputs=self.dis_input,
                                   outputs=self.dis_output,
                                   name='discriminator')

        # self.discriminator.compile(loss='binary_crossentropy',
        #                            optimizer=SGD())
        print self.discriminator.summary()


if __name__ == '__main__':
    model = Context_Encoder()
    a = np.ones(shape=(10, 3, 128, 128))
    b = model.encoder_decoder.predict(a)
    print b.shape
    # b = model.encoder_output
