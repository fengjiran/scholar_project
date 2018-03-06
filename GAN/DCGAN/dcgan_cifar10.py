import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.optimizers import RMSprop
from keras import backend as K
import numpy as np
from PIL import Image
from load_cifar10 import load_cifar_10


def generator_model():
    """Generator."""
    model = Sequential()
    model.add(Dense(units=256 * 8 * 8, input_dim=100))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Reshape((256, 8, 8)))
    model.add(Dropout(0.5))

    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2DTranspose(128, (5, 5), padding='same', data_format=K.image_data_format()))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1, momentum=0.9))

    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2DTranspose(64, (5, 5), padding='same', data_format=K.image_data_format()))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1, momentum=0.9))

    model.add(Conv2DTranspose(32, (5, 5), padding='same', data_format=K.image_data_format()))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1, momentum=0.9))

    model.add(Conv2DTranspose(3, (5, 5), padding='same', data_format=K.image_data_format()))
    model.add(Activation('tanh'))
    print model.summary()

    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(filters=64,
                     kernel_size=(5, 5),
                     strides=(2, 2),
                     padding='same',
                     input_shape=(3, 32, 32),
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

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid'))

    print model.summary()
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    # discriminator.trainable = False
    model.add(discriminator)
    print model.name

    return model


def combine_images(generated_images):
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


def train(BATCH_SIZE):
    """Train the model."""
    (X_train, y_train), (X_test, y_test) = load_cifar_10()
    X_train = 2 * X_train - 1  # fall in the interval [-1, 1]

    discriminator = discriminator_model()
    generator = generator_model()
    discriminator.trainable = False
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

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

            discriminator.trainable = True

            print "batch %d g_loss : %f" % (index, g_loss)

        image = combine_images(generated_images)
        image = image * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save('./generated_cifar10/' + str(epoch) + ".png")

        generator.save_weights('generator_cifar10', True)
        discriminator.save_weights('discriminator_cifar10', True)


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
    # elif args.mode == "generate":
    #     generate(BATCH_SIZE=args.batch_size, nice=args.nice)
