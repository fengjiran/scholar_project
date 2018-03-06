from __future__ import division
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l1_l2
from keras.datasets import mnist
from PIL import Image
import h5py


class GAN(object):
    """Simple MLP GAN."""

    def __init__(self,
                 latent_dim=100,
                 hidden_dim=1024,
                 image_shape=(28, 28),
                 batch_size=128,
                 epochs=100):
        """Network."""
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.epochs = epochs

        generator_layer = [Dense(units=int(self.hidden_dim / 4),
                                 input_dim=int(self.latent_dim),
                                 kernel_regularizer=l1(1e-5)),
                           BatchNormalization(),
                           LeakyReLU(0.2),
                           Dense(units=int(self.hidden_dim / 2),
                                 kernel_regularizer=l1(1e-5)),
                           BatchNormalization(),
                           LeakyReLU(0.2),
                           Dense(units=self.hidden_dim,
                                 kernel_regularizer=l1(1e-5)),
                           BatchNormalization(),
                           LeakyReLU(0.2),
                           Dense(units=np.prod(self.image_shape),
                                 kernel_regularizer=l1(1e-5),
                                 activation='tanh')]

        discirminator_layer = [Dense(units=self.hidden_dim,
                                     input_dim=np.prod(self.image_shape),
                                     kernel_regularizer=l1_l2(1e-5, 1e-5)),
                               #    BatchNormalization(),
                               LeakyReLU(0.2),
                               Dropout(0.5),
                               Dense(units=int(self.hidden_dim / 2),
                                     kernel_regularizer=l1_l2(1e-5, 1e-5)),
                               #    BatchNormalization(),
                               LeakyReLU(0.2),
                               Dropout(0.5),
                               Dense(units=int(self.hidden_dim / 4),
                                     kernel_regularizer=l1_l2(1e-5, 1e-5)),
                               #    BatchNormalization(),
                               LeakyReLU(0.2),
                               Dropout(0.5),
                               Dense(units=1,
                                     kernel_regularizer=l1_l2(1e-5, 1e-5),
                                     activation='sigmoid')]

        self.generator = Sequential(generator_layer)
        self.discriminator = Sequential(discirminator_layer)

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
        shape = generated_images.shape[1:]
        image = np.zeros((height * shape[0], width * shape[1]),
                         dtype=generated_images.dtype)

        for ind, img in enumerate(generated_images):
            i = int(ind / width)
            j = ind % width
            image[i * shape[0]:(i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1]] = img

        return image

    def train(self):
        """Train the model."""
        (X_train, y_train), (X_test, y_test) = mnist.load_data('/home/richard/datasets/mnist.npz')
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = X_train.astype(np.float32) / 255.0
        # print X_train.shape
        # X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[1]))

        d_optim = SGD(lr=1e-3, momentum=0.9, nesterov=True)
        # g_optim = Adam(lr=1e-5)
        g_optim = SGD(lr=1e-4, momentum=0.9, nesterov=True)

        self.gan_model.compile(loss='binary_crossentropy', optimizer=g_optim)

        self.discriminator.trainable = True

        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

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
                print 'Batch {} d_loss: {}'.format(ind, d_loss)

                for i in range(noise.shape[0]):
                    noise[i] = np.random.normal(loc=0.0,
                                                scale=1.0,
                                                size=self.latent_dim)

                # noise = np.random.normal(loc=0.0,
                #                          scale=1.0,
                #                          size=(self.batch_size, self.latent_dim))

                self.discriminator.trainable = False
                g_loss = self.gan_model.train_on_batch(noise, [1] * self.batch_size)
                self.discriminator.trainable = True

                print 'Batch {} g_loss: {}'.format(ind, g_loss)

            new_shape = (generated_images.shape[0],) + self.image_shape
            generated_images = np.reshape(generated_images, new_shape)
            image = self.combine_images(generated_images)

            image = image * 127.5 + 127.5
            Image.fromarray(image.astype(np.uint8)).save(str(epoch) + '.png')

            self.generator.save_weights('generator.h5', True)
            self.discriminator.save_weights('discriminator.h5', True)

    def generate(self, nice=False):
        self.generator.load_weights('generator.h5')
        if nice:
            self.discriminator.load_weights('discriminator.h5')
            noise = np.zeros((self.batch_size * 20, self.latent_dim))
            for i in range(noise.shape[0]):
                noise[i] = np.random.normal(loc=0.0,
                                            scale=1.0,
                                            size=self.latent_dim)

            generated_image = self.generator.predict(noise, verbose=1)
            d_pret = self.discriminator.predict(generated_image, verbose=1)

            index = np.arange(0, noise.shape[0])
            index.resize((noise.shape[0], 1))

            pre_with_index = list(np.append(d_pret, index, axis=1))
            pre_with_index.sort(key=lambda x: x[0], reverse=True)

            nice_image = np.zeros((self.batch_size, np.prod(self.image_shape)),
                                  dtype=np.float32)

            for i in range(self.batch_size):
                idx = int(pre_with_index[i][1])
                nice_image[i] = generated_image[idx]
            new_shape = (nice_image.shape[0],) + self.image_shape
            nice_image = np.reshape(nice_image, new_shape)
            image = self.combine_images(nice_image)
        else:
            noise = np.zeros((self.batch_size, self.latent_dim))
            for i in range(noise.shape[0]):
                noise[i] = np.random.normal(loc=0.0,
                                            scale=1.0,
                                            size=self.latent_dim)
            generated_image = self.generator.predict(noise, verbose=1)
            new_shape = (generated_image.shape[0],) + self.image_shape
            generated_image = np.reshape(generated_image, new_shape)
            # print generated_image.shape
            image = self.combine_images(generated_image)

        image = image * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save('generated_image.png')


if __name__ == '__main__':
    model = GAN(latent_dim=100,
                hidden_dim=1024,
                image_shape=(28, 28),
                batch_size=128,
                epochs=100)
    # model.train()
    model.generate(True)
    print 'done!'
