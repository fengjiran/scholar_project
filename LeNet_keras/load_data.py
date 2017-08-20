﻿import cPickle
import gzip
import os
import sys
import time

import pylab
from PIL import Image

import numpy as np
# import theano
# import theano.tensor as T


def load_mnist(dataset='/home/richard/datasets/mnist.pkl.gz'):
    '''Load the dataset.
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST) 
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # which row's correspond to an example.
    # target is a numpy.ndarray of 1 dimension (a vector) that have
    # the same length as the number of rows in the input.
    # It should give the target to the example with the same index
    # in the input.

    # def shared_dataset(data_xy, borrow = True):
    #     """Function that loads the dataset into shared variables.

    #     The reason we store our dataset in shared variables is to allow
    #     Theano to copy it into the GPU memory (when code is run on GPU).
    #     Since copying data into the GPU is slow, copying a minibatch
    #     everytime is needed (the default behaviour if the data is not in
    #     a shared variables) would lead to a large decrease in performance.
    #     """
    #     data_x, data_y = data_xy
    #     shared_x = theano.shared(np.asarray(data_x,
    #                                         dtype = theano.config.floatX),
    #                              borrow = borrow)
    #     shared_y = theano.shared(np.asarray(data_y,
    #                                         dtype = theano.config.floatX),
    #                              borrow = borrow)

    #     # When storing data on the GPU it has to be stored as floats
    #     # therefore we will store the labels as ``floatX`` as well
    #     # (``shared_y`` does exactly that). But during our computations
    #     # we need them as ints (we use labels as index, and if they are
    #     # floats it doesn't make sense) therefore instead of returning
    #     # ``shared_y`` we will have to cast it to int. This little hack
    #     # lets ous get around this issue

    #     return shared_x, T.cast(shared_y, 'int32')

    # train_set_x, train_set_y = shared_dataset(train_set)
    # valid_set_x, valid_set_y = shared_dataset(valid_set)
    # test_set_x, test_set_y = shared_dataset(test_set)

    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


if __name__ == '__main__':
    train_set, valid_set, test_set = load_mnist()
    train_set_x, train_set_y = train_set

    img_shape = (28, 28)
    print train_set_x.get_value(borrow=True).shape
    print train_set_x.get_value(borrow=True)[0].reshape(img_shape)

    for i in range(100):
        img = train_set_x.get_value(borrow=True)[i].reshape(img_shape)
        pylab.subplot(10, 10, i + 1)
        pylab.axis('off')
        pylab.imshow(img)

    pylab.show()
