"""https://github.com/jacobgil/keras-dcgan."""
import argparse
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Flatten
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras import backend as K
import numpy as np
from PIL import Image


def generator_model():
    """Generator."""
    model = Sequential()
    model.add(Dense(units=256 * 7 * 7, input_dim=100))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Reshape((256, 7, 7)))
    model.add(Dropout(0.5))

    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2DTranspose(128, (5, 5), padding='same', data_format=K.image_data_format()))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2DTranspose(64, (5, 5), padding='same', data_format=K.image_data_format()))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(32, (5, 5), padding='same', data_format=K.image_data_format()))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(1, (5, 5), padding='same', data_format=K.image_data_format()))
    model.add(Activation('tanh'))
    print model.summary()

    return model

# def generator_model():
#     """Generator."""
#     model = Sequential()
#     model.add(Dense(units=1024, input_dim=100, activation='tanh'))
#     model.add(Dense(units=128 * 7 * 7))
#     model.add(BatchNormalization())
#     model.add(Activation('tanh'))
#     model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7, )))
#     model.add(UpSampling2D(size=(2, 2)))
#     model.add(Conv2D(64, (5, 5), padding='same', activation='tanh'))
#     model.add(UpSampling2D(size=(2, 2)))
#     model.add(Conv2D(1, (5, 5), padding='same', activation='tanh'))

#     return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(filters=64,
                     kernel_size=(5, 5),
                     strides=(2, 2),
                     padding='same',
                     input_shape=(1, 28, 28),
                     activation=LeakyReLU(0.2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64 * 2,
                     kernel_size=(5, 5),
                     strides=(2, 2),
                     padding='same',
                     activation=LeakyReLU(0.2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64 * 4,
                     kernel_size=(5, 5),
                     strides=(2, 2),
                     padding='same',
                     activation=LeakyReLU(0.2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64 * 8,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     padding='same',
                     activation=LeakyReLU(0.2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    print model.summary()
    return model

# def discriminator_model():
#     model = Sequential()
#     model.add(Conv2D(filters=64,
#                      kernel_size=(5, 5),
#                      padding='same',
#                      input_shape=(1, 28, 28),
#                      activation='tanh'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(128, (5, 5), activation='tanh'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(1024, activation='tanh'))
#     model.add(Dense(1, activation='sigmoid'))
#     return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    # discriminator.trainable = False
    model.add(discriminator)
    print model.name
    # print len(model.layers)
    # print model.layers[0].name, model.layers[0].trainable
    # print model.layers[1].name, model.layers[1].trainable
    # print model.get_layer(index=0).name
    # print model.get_layer(index=1).name
    # print discriminator.trainable
    return model


def combine_images(generated_images):
    """Combine the images."""
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]

    return image


def train(BATCH_SIZE):
    (X_train, y_train), (X_test, y_test) = mnist.load_data('/home/richard/datasets/mnist.npz')
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    discriminator = discriminator_model()
    generator = generator_model()
    discriminator.trainable = False
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

    # d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    # g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    d_optim = RMSprop(lr=8e-4, clipvalue=1.0, decay=6e-8)
    g_optim = RMSprop(lr=4e-4, clipvalue=1.0, decay=3e-8)

    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):

            noise = np.random.normal(loc=0, scale=0.02, size=(BATCH_SIZE, 100))

            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            # print image_batch.shape

            generated_images = generator.predict(noise, verbose=0)
            # print generated_images.shape

            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            print "batch %d d_loss : %f" % (index, d_loss)
            noise = np.random.normal(loc=0, scale=0.02, size=(BATCH_SIZE, 100))

            discriminator.trainable = False

            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * BATCH_SIZE)

            # print discriminator_on_generator.layers[0].trainable
            # print discriminator_on_generator.layers[1].trainable

            discriminator.trainable = True

            print "batch %d g_loss : %f" % (index, g_loss)

        image = combine_images(generated_images)
        image = image * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save(str(epoch) + ".png")

        generator.save_weights('generator', True)
        discriminator.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE * 20, 100))
        for i in range(BATCH_SIZE * 20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE * 20)
        index.resize((BATCH_SIZE * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) + (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save("generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # generator_model()
    # discriminator_model()
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
