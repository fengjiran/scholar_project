from __future__ import division
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, LeakyReLU
# from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l1_l2
from keras.datasets import mnist
from PIL import Image


def generator_model():
    """Generator."""
    generator_layer = [Dense(units=256,
                             input_dim=100,
                             kernel_regularizer=l1(1e-5)),
                       BatchNormalization(),
                       LeakyReLU(0.2),
                       Dense(units=512,
                             kernel_regularizer=l1(1e-5)),
                       BatchNormalization(),
                       LeakyReLU(0.2),
                       Dense(units=1024,
                             kernel_regularizer=l1(1e-5)),
                       BatchNormalization(),
                       LeakyReLU(0.2),
                       Dense(units=28 * 28,
                             kernel_regularizer=l1(1e-5),
                             activation='sigmoid')]

    model = Sequential(generator_layer)

    print 'The generator_model summary:'
    print model.summary()
    return model


def discriminator_model():
    discirminator_layer = [Dense(units=1024,
                                 input_shape=(784,),
                                 kernel_regularizer=l1_l2(1e-5, 1e-5)),
                           #    BatchNormalization(),
                           LeakyReLU(0.2),
                           Dropout(0.5),
                           Dense(units=512,
                                 kernel_regularizer=l1_l2(1e-5, 1e-5)),
                           #    BatchNormalization(),
                           LeakyReLU(0.2),
                           Dropout(0.5),
                           Dense(units=256,
                                 kernel_regularizer=l1_l2(1e-5, 1e-5)),
                           #    BatchNormalization(),
                           LeakyReLU(0.2),
                           Dropout(0.5),
                           Dense(units=1,
                                 kernel_regularizer=l1_l2(1e-5, 1e-5),
                                 activation='sigmoid')]

    model = Sequential(discirminator_layer)
    print 'The discriminator_model summary:'
    print model.summary()

    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)

    return model


def combine_images(generated_images):
    """Combine the images."""
    num = generated_images.shape[0]
    width = int(np.sqrt(num))
    height = int(np.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]

    return image


def train(BATCH_SIZE=128):
    (X_train, y_train), (X_test, y_test) = mnist.load_data('/home/richard/datasets/mnist.npz')
    # (X_train, y_train), (X_test, y_test) = mnist.load_data(r'E:\deeplearning_experiments\datasets\mnist.npz')
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    # X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[1]))

    discriminator = discriminator_model()
    generator = generator_model()
    # discriminator.trainable = False
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

    d_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0001, momentum=0.9, nesterov=True)

    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):

            noise = np.random.uniform(low=-1, high=1, size=(BATCH_SIZE, 100))

            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]

            # print 'image_batch shape {}'.format(image_batch.shape)

            generated_images = generator.predict(noise, verbose=0)
            # print 'generated_images shape {}'.format(generated_images.shape)

            X_batch = np.concatenate((image_batch, generated_images))
            # print 'X_batch shape {}'.format(X_batch.shape)

            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            # discriminator.fit(x=X_batch, y=y, batch_size=256, epochs=1)
            d_loss = discriminator.train_on_batch(X_batch, y)

            print "batch %d d_loss : %f" % (index, d_loss)
            noise = np.random.uniform(low=-1, high=1, size=(BATCH_SIZE, 100))

            discriminator.trainable = False

            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * BATCH_SIZE)

            discriminator.trainable = True

            print "batch %d g_loss : %f" % (index, g_loss)

        new_shape = (generated_images.shape[0], 1, 28, 28)
        generated_images = np.reshape(generated_images, new_shape)
        image = combine_images(generated_images)
        image = image * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save(str(epoch) + ".png")

        generator.save_weights('generator', True)
        discriminator.save_weights('discriminator', True)


if __name__ == "__main__":
    train()
