import os
# import cPickle
import numpy as np
# import scipy
import pylab
from keras.datasets.cifar import load_batch
from keras.utils.np_utils import to_categorical
# from keras.utils.data_utils import get_file
from keras import backend as K


def load_cifar_100(datapath='/home/richard/datasets/cifar-100-python',
                   label_mode='fine'):
    """Load CIFAR 100 dataset.

    # Arguments
        label_mode: one of "fine", "coarse".
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    # Raises
        ValueError: in case of invalid `label_mode`.
    """
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('label_mode must be one of "fine" and "coarse".')

    fpath = os.path.join(datapath, 'train')
    X_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(datapath, 'test')
    X_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    y_train = to_categorical(y_train, num_classes=100)
    y_test = to_categorical(y_test, num_classes=100)

    if K.image_data_format() == 'channels_last':
        X_train = X_train.transpose(0, 2, 3, 1)
        X_test = X_test.transpose(0, 2, 3, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    train, test = load_cifar_100(datapath=r'E:\deeplearning_experiments\datasets\cifar-100-python')
    X_train, y_train = train
    X_test, y_test = test

    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape

    print X_train[0].max()
    print X_train[0].min()
    print X_train[0].mean()

    for i in range(100):
        img = X_train[i].transpose(1, 2, 0)
        pylab.subplot(10, 10, i + 1)
        pylab.imshow(img)
        pylab.axis('off')
    pylab.show()
