import os
import cPickle
import numpy as np
import scipy.io as sio

from keras.utils import to_categorical


def load_cifar_10(datapath='/home/richard/datasets/cifar-10-batches-py'):
    dtype = 'uint8'
    ntrain = 50000
    # nvalid = 0
    ntest = 10000

    img_shape = (3, 32, 32)
    image_size = np.prod(img_shape)
    # n_classes = 10
    # label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    fnames = ['data_batch_%i' % i for i in range(1, 6)]
    # or
    # fnames = ['data_batch_{0}'.format(i) for i in range(1, 6)]

    train_set_x = np.zeros((ntrain, image_size), dtype='float32')
    train_set_y = np.zeros((ntrain,), dtype=dtype)

    test_set_x = np.zeros((ntest, image_size), dtype='float32')
    test_set_y = np.zeros((ntest,), dtype=dtype)

    # load train data
    for i, name in zip(range(5), fnames):
        fname = os.path.join(datapath, name)
        if not os.path.exists(fname):
            raise IOError(fname + "was not found. You probably need to "
                          "download the CIFAR-10 dataset "
                          "manually from "
                          "http://www.cs.utoronto.ca/~kriz/cifar.html")

        fo = open(fname, 'rb')
        dicts = cPickle.load(fo)

        # a = dicts['data'].copy(True)
        a = np.copy(dicts['data']).astype('float32')
        # a /= 255.0
        train_set_x[i * 10000:(i + 1) * 10000, :] = a
        train_set_y[i * 10000:(i + 1) * 10000] = dicts['labels']
        fo.close()

    fo = open(os.path.join(datapath, 'test_batch'), 'rb')
    dicts = cPickle.load(fo)
    a = np.copy(dicts['data']).astype('float32')
    # a /= 255.0

    test_set_x[0:10000, :] = a
    test_set_y[0:10000] = dicts['labels']

    fo.close()

    train_set_y = np.reshape(train_set_y, (len(train_set_y), 1))
    test_set_y = np.reshape(test_set_y, (len(test_set_y), 1))

    train_set_y = to_categorical(train_set_y, num_classes=10)
    test_set_y = to_categorical(test_set_y, num_classes=10)

    train_set_x = np.reshape(train_set_x, (ntrain, 3, 32, 32))
    test_set_x = np.reshape(test_set_x, (ntest, 3, 32, 32))

    return [(train_set_x, train_set_y), (test_set_x, test_set_y)]


def load_cifar_100(datapath='/home/richard/datasets/cifar-100-python'):

    ntrain = 50000
    # nvalid = 0
    ntest = 10000

    with open(os.path.join(datapath, 'train'), 'rb') as f:
        dicts = cPickle.load(f)
        train_set_x = dicts['data'].astype('float32')
        train_set_y = np.array(dicts['fine_labels'])

        train_set_x = np.reshape(train_set_x, (ntrain, 3, 32, 32))
        train_set_y = np.reshape(train_set_y, (ntrain, 1))

        train_set_y = to_categorical(train_set_y, num_classes=100)

    with open(os.path.join(datapath, 'test'), 'rb') as f:
        dicts = cPickle.load(f)
        test_set_x = dicts['data'].astype('float32')
        test_set_y = np.array(dicts['fine_labels'])

        test_set_x = np.reshape(test_set_x, (ntest, 3, 32, 32))
        test_set_y = np.reshape(test_set_y, (ntest, 1))

        test_set_y = to_categorical(test_set_y, num_classes=100)

    return [(train_set_x, train_set_y), (test_set_x, test_set_y)]


def load_svhn(datapath='/home/richard/datasets/SVHN'):

    train_set = sio.loadmat(os.path.join(datapath, 'train_32x32.mat'))
    test_set = sio.loadmat(os.path.join(datapath, 'test_32x32.mat'))

    train_set_x = train_set['X'].astype('float32')  # (32, 32, 3, 73257)
    train_set_y = train_set['y'].flatten()

    test_set_x = test_set['X'].astype('float32')  # (32, 32, 3, 26032)
    test_set_y = test_set['y'].flatten()

    train_set_x = np.transpose(train_set_x, (3, 2, 0, 1))
    test_set_x = np.transpose(test_set_x, (3, 2, 0, 1))

    train_set_x /= 255.0
    test_set_x /= 255.0

    train_set_y[train_set_y == 10] = 0
    test_set_y[test_set_y == 10] = 0

    train_set_y = to_categorical(train_set_y, num_classes=10)
    test_set_y = to_categorical(test_set_y, num_classes=10)

    return [(train_set_x, train_set_y), (test_set_x, test_set_y)]


if __name__ == '__main__':
    train_set, test_set = load_svhn()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set

    print train_set_x.shape
    print test_set_x.shape
    print train_set_y.shape
    print test_set_y.shape
    print train_set_x.max()
