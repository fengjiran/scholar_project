from __future__ import division
import numpy as np
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, LeakyReLU, concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l1_l2
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from PIL import Image
import h5py


class CGAN(object):
    """Simple MLP CGAN."""

    def __init__(self,
                 latent_dim=100,
                 image_shape=(28, 28),
                 batch_size=100,
                 epochs=100):
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.epochs = epochs

        # Construct the generator
        p_z = Input(shape=(100,))
        x = Dense(units=200,
                  kernel_regularizer=l1(1e-5))(p_z)
        x = LeakyReLU(0.2)(x)

        condition_y = Input(shape=(10,))
        y = Dense(units=1000,
                  kernel_regularizer=l1(1e-5))(condition_y)
        y = LeakyReLU(0.2)(y)

        merge_xy = concatenate([x, y], axis=1)

        g_outputs = Dense(units=784,
                          activation='tanh',
                          kernel_regularizer=l1(1e-5))(merge_xy)
        self.generator = Model(inputs=[p_z, condition_y],
                               outputs=g_outputs)

        # Construct the discriminator
        d_x = Input(shape=(784,))
        d_condition_y = Input(shape=(10,))
        d_input = concatenate([d_x, d_condition_y], axis=1)

        d_input = Dense(units=128,
                        kernel_regularizer=l1(1e-5))(d_input)

        d_input = LeakyReLU(0.2)(d_input)
        d_output = Dense(units=1,
                         activation='sigmoid',
                         kernel_regularizer=l1(1e-5))(d_input)
        self.discriminator = Model(inputs=[d_x, d_condition_y],
                                   outputs=d_output)

        print self.generator.summary()
        print self.discriminator.summary()

    def train(self):
        d_optim = Adam(lr=2e-4, beta_1=0.5)
        g_optim = Adam(lr=2e-4, beta_1=0.5)

        self.discriminator.compile(optimizer=d_optim,
                                   loss='binary_crossentropy')

        self.generator.compile(optimizer=g_optim,
                               loss='binary_crossentropy')

        latent = Input(shape=(self.latent_dim,))
        g_condition = Input(shape=(10,))
        d_condition = Input(shape=(10,))

        # Get the fake image
        fake = self.generator([latent, g_condition])

        # we only want to be able to train generation for the combined model
        self.discriminator.trainable = False
        d_output = self.discriminator([fake, d_condition])

        combined_model = Model(inputs=[latent, g_condition, d_condition],
                               outputs=d_output)

        combined_model.compile(optimizer=g_optim,
                               loss='binary_crossentropy')

        (X_train, y_train), (X_test, y_test) = mnist.load_data('/home/richard/datasets/mnist.npz')
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[1]))

        condition = []
        for i in range(10):
            condition.extend([i] * 10)

        condition = np.asarray(condition)

        # one-hot encode
        condition = to_categorical(condition, 10)

        for epoch in range(self.epochs):
            print 'Epoch {} of {}'.format(epoch + 1, self.epochs)
            num_batches = int(X_train.shape[0] / self.batch_size)

            for index in range(num_batches):
                noise = np.random.normal(loc=0.0,
                                         scale=1.0,
                                         size=(self.batch_size, self.latent_dim))

                image_batch = X_train[index * self.batch_size:(index + 1) * self.batch_size]
                generated_images = self.generator.predict([noise, condition], verbose=0)

                X = np.concatenate((image_batch, generated_images))


if __name__ == '__main__':
    model = CGAN()
