import numpy as np
import os
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
from keras.datasets.cifar import load_batch
from scipy import linalg


def load_data():
    """Loads CIFAR10 dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = '/home/richard/datasets/cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin)

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    # print y_train.shape
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def global_contrast_normalization(x, samplewise_center=True, samplewise_std_normalization=True):
    data_format = K.image_data_format()
    # channel_axis = 0
    if data_format == 'channels_first':
        channel_axis = 1
        row_axis = 2
        col_axis = 3
    if data_format == 'channels_last':
        channel_axis = 3
        row_axis = 1
        col_axis = 2

    # img_channel_axis = channel_axis - 1

    if samplewise_center:

        x -= np.mean(x, axis=channel_axis, keepdims=True)

    if samplewise_std_normalization:
        x /= (np.std(x, axis=channel_axis, keepdims=True) + 1e-7)

    return x


def zca(x):
    flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
    sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
    u, s, _ = linalg.svd(sigma)
    principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T)

    whitex = np.dot(flat_x, principal_components)

    x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))

    return x


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = np.asarray(x_train, dtype='float32')
    x_test = np.asarray(x_test, dtype='float32')

    x_train = global_contrast_normalization(x_train)
    a = x_train[0].reshape(3072)
    print a.mean()
    # x_train = zca(x_train)
    print x_train.shape
    print x_test.shape
    # print x_train.size
