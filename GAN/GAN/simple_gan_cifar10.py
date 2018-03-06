from __future__ import division
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, Dense, Activation, SpatialDropout2D
from keras.layers import LeakyReLU, Reshape, UpSampling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.regularizers import l1_l2
# from keras.datasets import mnist
from PIL import Image

from load_cifar10 import load_cifar_10


class GAN(object):
    """Simple GAN for cifar10."""

    def __init__(self,
                 latent_dim=100,
                 image_shape=(3, 32, 32),
                 batch_size=128,
                 epochs=100,
                 k_g=1,
                 k_d=1):
        """Network."""
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_g = k_g
        self.k_d = k_d

        nch = 256
        h = 5

        generator_layer = [Dense(units=nch * 4 * 4,
                                 input_dim=self.latent_dim,
                                 kernel_regularizer=l1_l2(1e-7, 1e-7)),
                           BatchNormalization(),
                           Reshape((nch, 4, 4)),

                           Conv2D(filters=int(nch / 2),
                                  kernel_size=(h, h),
                                  padding='same',
                                  kernel_regularizer=l1_l2(1e-7, 1e-7)),
                           BatchNormalization(axis=1),
                           LeakyReLU(0.2),
                           UpSampling2D(size=(2, 2)),

                           Conv2D(filters=int(nch / 2),
                                  kernel_size=(h, h),
                                  padding='same',
                                  kernel_regularizer=l1_l2(1e-7, 1e-7)),
                           BatchNormalization(axis=1),
                           LeakyReLU(0.2),
                           UpSampling2D(size=(2, 2)),

                           Conv2D(filters=int(nch / 4),
                                  kernel_size=(h, h),
                                  padding='same',
                                  kernel_regularizer=l1_l2(1e-7, 1e-7)),
                           BatchNormalization(axis=1),
                           LeakyReLU(0.2),
                           UpSampling2D(size=(2, 2)),

                           Conv2D(filters=3,
                                  kernel_size=(h, h),
                                  padding='same',
                                  kernel_regularizer=l1_l2(1e-7, 1e-7)),
                           Activation('tanh')]

        discriminator_layer = [Conv2D(filters=int(nch / 4),
                                      input_shape=(self.image_shape),
                                      kernel_size=(h, h),
                                      padding='same',
                                      kernel_regularizer=l1_l2(1e-7, 1e-7)),
                               SpatialDropout2D(0.5),
                               MaxPooling2D(pool_size=(2, 2)),
                               LeakyReLU(0.2),

                               Conv2D(filters=int(nch / 2),
                                      kernel_size=(h, h),
                                      padding='same',
                                      kernel_regularizer=l1_l2(1e-7, 1e-7)),
                               SpatialDropout2D(0.5),
                               MaxPooling2D(pool_size=(2, 2)),
                               LeakyReLU(0.2),

                               Conv2D(filters=nch,
                                      kernel_size=(h, h),
                                      padding='same',
                                      kernel_regularizer=l1_l2(1e-7, 1e-7)),
                               SpatialDropout2D(0.5),
                               MaxPooling2D(pool_size=(2, 2)),
                               LeakyReLU(0.2),

                               Conv2D(filters=1,
                                      kernel_size=(h, h),
                                      padding='same',
                                      kernel_regularizer=l1_l2(1e-7, 1e-7)),
                               AveragePooling2D(pool_size=(4, 4), padding='valid'),
                               Flatten(),
                               Activation('sigmoid')]

        self.generator = Sequential(generator_layer)
        self.discriminator = Sequential(discriminator_layer)

        print self.generator.summary()
        print self.discriminator.summary()

        self.gan_model = Sequential()
        self.gan_model.add(self.generator)
        self.discriminator.trainable = False
        self.gan_model.add(self.discriminator)

    def combine_images(self, generated_images):
        """Combine the images."""
        num = generated_images.shape[0]
        width = int(np.sqrt(num))
        height = int(np.ceil(float(num) / width))
        shape = generated_images.shape[2:]
        image = np.zeros((height * shape[0], width * shape[1], 3), dtype=generated_images.dtype)

        for index, img in enumerate(generated_images):
            i = int(index / width)
            j = index % width
            img = img.transpose(1, 2, 0)
            image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img

        return image

    def train(self):
        """Train the model."""
        (X_train, y_train), (X_test, y_test) = load_cifar_10()
        X_train = 2 * X_train - 1  # fall in the interval [-1, 1]

        d_optim = SGD(lr=1e-4, momentum=0.9, nesterov=True)
        g_optim = SGD(lr=1e-4, momentum=0.9, nesterov=True)

        # d_optim = Adam(lr=1e-4, decay=1e-5)
        # g_optim = Adam(lr=1e-3, decay=1e-5)
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=g_optim)

        self.gan_model.compile(loss='binary_crossentropy',
                               optimizer=g_optim)

        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=d_optim)

        noise = np.zeros((self.batch_size, self.latent_dim))

        for epoch in range(self.epochs):
            for ind in range(int(X_train.shape[0] / self.batch_size)):
                for i in range(noise.shape[0]):
                    noise[i] = np.random.normal(loc=0.0,
                                                scale=1.0,
                                                size=self.latent_dim)

                image_batch = X_train[ind * self.batch_size:(ind + 1) * self.batch_size]

                generated_images = self.generator.predict(noise, verbose=0)

                X = np.concatenate((image_batch, generated_images))
                y = [1] * self.batch_size + [0] * self.batch_size
                d_loss = self.discriminator.train_on_batch(X, y)
                print 'Batch {} g_loss: {}'.format(ind, d_loss)
                # history_d = self.discriminator.fit(X, y,
                #                                    batch_size=X.shape[0],
                #                                    epochs=self.k_d)

                for i in range(noise.shape[0]):
                    noise[i] = np.random.normal(loc=0.0,
                                                scale=1.0,
                                                size=self.latent_dim)

                X = noise
                y = [1] * self.batch_size
                self.discriminator.trainable = False
                g_loss = self.gan_model.train_on_batch(X, y)
                print 'Batch {} g_loss: {}'.format(ind, g_loss)
                # history_g = self.gan_model.fit(X, y,
                #                                batch_size=X.shape[0],
                #                                epochs=self.k_g)

                self.discriminator.trainable = True

            image = self.combine_images(generated_images)
            image = image * 127.5 + 127.5
            Image.fromarray(image.astype(np.uint8)).save(str(epoch) + '.png')

        self.generator.save_weights('generator_cifar10.h5', True)
        self.discriminator.save_weights('discriminator_cifar10.h5', True)


if __name__ == '__main__':
    model = GAN(latent_dim=100,
                image_shape=(3, 32, 32),
                batch_size=128,
                epochs=100,
                k_g=1,
                k_d=1)

    model.train()

    print 'Done!'
