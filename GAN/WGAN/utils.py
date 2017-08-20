import math
import numpy as np
from keras import backend as K
from keras.datasets import mnist


def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)


def get_processed_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data('/home/richard/datasets/mnist.npz')
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])

    return (X_train, y_train), (X_test, y_test)


def sample_unit_gaussian(num_rows=1, dimension=1):
    return np.random.normal(size=(num_rows, dimension))


def get_batch(data, batch_size=1):
    return data[np.random.randint(data.shape[0], size=batch_size)]


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
