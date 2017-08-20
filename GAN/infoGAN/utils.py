import time
import math
import numpy as np
from keras.datasets import mnist
from keras import backend as K

EPSILON = 1e-8


def disc_mutual_info_loss(c_disc, aux_dist):
    """Mutual information lower bound loss for discrete distribution."""
    reg_disc_dim = aux_dist.get_shape().as_list()[-1]
    cross_ent = - K.mean(K.sum(K.log(aux_dist + EPSILON) * c_disc, axis=1))
    ent = - K.mean(K.sum(K.log(1. / reg_disc_dim + EPSILON) * c_disc, axis=1))  # H(c)
    return ent - cross_ent


def get_processed_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data('/home/richard/datasets/mnist.npz')
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])

    return (X_train, y_train), (X_test, y_test)


def sample_unit_gaussian(num_rows=1, dimension=1):
    return np.random.normal(size=(num_rows, dimension))


def sample_categorical(num_rows=1, num_categories=2):
    sample = np.zeros(shape=(num_rows, num_categories))
    sample[np.arange(num_rows), np.random.randint(num_categories, size=num_rows)] = 1
    return sample


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
# def sample_c(m):
#     return np.random.multinomial(1, 10 * [0.1], size=m)


if __name__ == '__main__':
    num_rows = 20
    num_categories = 10
    c_disc = sample_categorical(num_rows, num_categories)

    # print sample_c(5)
