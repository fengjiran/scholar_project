"""
http://www.cs.toronto.edu/~kriz/cifar.html
https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/datasets/cifar10.py

"""

import os
import sys

import cPickle
import gzip
import numpy as np
import scipy
import pylab
from PIL import Image

import theano
import theano.tensor as T
from sklearn.feature_extraction.image import extract_patches_2d


def load_cifar_10(datapath='/home/richard/datasets/cifar-10-batches-py'):
    '''
    '''
    dtype = 'uint8'
    ntrain = 50000
    nvalid = 0
    ntest = 10000

    img_shape = (3, 32, 32)
    image_size = np.prod(img_shape)
    n_classes = 10
    label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    fnames = ['data_batch_%i' % i for i in range(1, 6)]
    # or
    #fnames = ['data_batch_{0}'.format(i) for i in range(1, 6)]

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
        dict = cPickle.load(fo)

        #a = dict['data'].copy(True)
        a = np.copy(dict['data']).astype('float32')
        a /= 256
        train_set_x[i * 10000:(i + 1) * 10000, :] = a
        train_set_y[i * 10000:(i + 1) * 10000] = dict['labels']
        fo.close()

    fo = open(os.path.join(datapath, 'test_batch'), 'rb')
    dict = cPickle.load(fo)
    a = np.copy(dict['data']).astype('float32')
    a /= 256

    test_set_x[0:10000, :] = a
    test_set_y[0:10000] = dict['labels']
    fo.close()

    # def shared(data_xy):
    #    """
    #    Place the data into shared variables. This allows Theano to copy
    #    the data to the GPU, if one is available.
    #    """

    #    data_x, data_y = data_xy
    #    shared_x = theano.shared(np.asarray(data_x,
    #                                        dtype = theano.config.floatX),
    #                             borrow = True)
    #    shared_y = theano.shared(np.asarray(data_y,
    #                                        dtype = theano.config.floatX),
    #                             borrow = True)
    #    return shared_x, T.cast(shared_y, 'int32')

    return [(train_set_x, train_set_y), (test_set_x, test_set_y)]


def extract_patches_for_classification(dataset, patch_width):
    '''
    '''
    n_samples = dataset.shape[0]
    patches = [extract_patches_2d(image=dataset[i],
                                  patch_size=(patch_width, patch_width)
                                  )
               for i in xrange(n_samples)
               ]

    patches = np.array(patches).reshape(-1, patch_width * patch_width * 3)

    return patches

# def extract_patches_for_classification(dataset, n_patches_per_image, patch_width):
#     '''
#     '''
#     n_samples = dataset.shape[0]
#     patches = [extract_patches_2d(image=dataset[i],
#                                   patch_size=(patch_width, patch_width),
#                                   max_patches=n_patches_per_image
#                                   )
#                for i in xrange(n_samples)
#                ]

#     patches = np.array(patches).reshape(-1, patch_width * patch_width * 3)

#     return patches


def extract_patches_for_pretrain(dataset, NSPL, patch_width):
    '''
    :param NSPL: number of samples to train layer
    :return: patches, array, shape = (n_patches, patch_height*patch_width) or
             (n_patches, patch_height*patch_width*n_channels).
             The collection of patches extracted from the image, where `n_patches`
             is either `max_patches` or the total number of patches that can be
             extracted.
    '''
    n_samples = dataset.shape[0]
    n_patches = NSPL / n_samples

    patches = [extract_patches_2d(image=dataset[i],
                                  patch_size=(patch_width, patch_width),
                                  max_patches=n_patches,
                                  random_state=i)
               for i in xrange(n_samples)]

    patches = np.array(patches).reshape(-1, patch_width * patch_width * 3)

    return patches


def shared_dataset_x(data_x, borrow=True):
    """Function that loads the dataset into shared variables.

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch 
    everytime is needed (the default behaviour if the data is not in
    a shared variables) would lead to a large decrease in performance.
    """

    shared_x = T._shared(
        np.asarray(data_x,
                   dtype=theano.config.floatX
                   ),
        borrow=borrow
    )
    #shared_x = theano.shared(data_x, borrow = borrow)

    return shared_x


def shared_dataset_y(data_y, borrow=True):
    shared_y = theano.shared(
        np.asarray(data_y,
                   dtype=theano.config.floatX
                   ),
        borrow=borrow
    )

    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue

    return T.cast(shared_y, 'int32')


def global_contrast_normalize(X, scale=1.0, subtract_mean=True, use_std=True, sqrt_bias=10.0, min_divisor=1e-8):
    """
    Global contrast normalizes by (optional) subtracting the mean across features and then normalizes by either the
    vector norm or the standard deviation (across features, for each example).

    Parameters
    --------------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and 
        features indexed on the second.

    scale : float, optional
        Multiply features by this const.

    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing.
        Defaults to 'True'.

    use_std : float, optional
        Normalize by the per-example standard deviation across features
        instead of the vector norm. Defaults to 'False'.

    sqrt_bias : float, optional
        Fudge factor added inside the aquare root. Defaults to 0.

    min_divisor : float, optional
        If the divisor for an example is less than this value,
        do not apply it. Defaults to '1e-8'.

    Returns
    ---------
    X : ndarray, 2-dimensional
        The contrast-normalized features

    Notes
    ---------
    'sqrt_bias = 10' and 'use_std = True' (and defaults for all other
    parameters) corresponds to the preprocessing used in [1].

    References
    ------------
    [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.

       http://ai.stanford.edu/~ang/papers/nipsdlufl10-AnalysisSingleLayerUnsupervisedFeatureLearning.pdf
       https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/preprocessing.py
    """
    assert X.ndim == 2, 'X.ndim must be 2'
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the
    # current object is the train, valid or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X -= mean[:, np.newaxis]  # Makes a copy
        #X -= mean[:, np.newaxis]
    else:
        X = X.copy()

    if use_std:
        # ddof = 1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0
        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, np.newaxis]   # Does not make a copy
    return np.asarray(X, dtype=np.float32)


def zca(X, epsilon=1e-6):
    """Computing the whiteniung and dewhitening matrices.
    Assuming the data points X have zero mean.

    Parameters
    -----------
    X : ndarray with shape [n_samples, n_features]
        The data used to compute the whitening and dewhitening
        matrices.

    epsilon : a small amount of regularization.
              When implementing ZCA whitening in practice,
              sometimes some of the eigenvalues will be 
              numerically close to 0, and thus the scaling 
              step where we divide by the square root of 
              eigenvalues would involve dividing by a value
              close to zero; this may cause the data to blow 
              up (take on large values) or otherwise be 
              numerically unstable. In practice, we therefore
              implement this scaling step using a small 
              amount of regularization, and add a small
              constant epsilon to the eigenvalues before
              taking their square root and inverse.

              When the data takes values around [-1, 1],
              a value of epsilon = 1e-5 might be typical.

              For the case of images, adding epsilon here 
              also has the effect of sightly smoothing
              (or low-pass filtering) the input images.
              This also has a desirable effect of removing
              aliasing artifacts caused by the way pixels 
              are laid out in an image, and can improve
              the featyres learned.

    Retures
    ----------
    whitening : ndarray, 2-dimensional
                The ZCA data

    References
    ------------
    http://ufldl.stanford.edu/wiki/index.php/Whitening
    https://github.com/mwv/zca/blob/master/zca/zca.py

    """
    # Computing the covariance matrix of X data points
    cov = np.dot(X.T, X) / (X.shape[0] - 1)

    # Computing the eigenvalues of covariance matrix
    # Because the covariance matrix is a symmeric
    # positive semi-definite matrix, it is more
    # numerically reliable to do this using the
    # svd function.
    # The matrix U will contain the eigenvectors
    # of covariance matrix (one eigenvector per
    # column, sorted in order from top to bottom
    # eigenvector), and the diagonal entries of
    # the matrix S will contain the corresponding
    # eigenvalues (also sorted in decreasing order).
    # Note: The svd function actually computes the
    # singular vectors and singular values of a matrix,
    # which for the special case of a symmetric positive
    # semi-definite matrix is equal to its eigenvectors
    # and eigenvalues.
    U, S, _ = scipy.linalg.svd(cov)
    s = np.sqrt(S.clip(epsilon))
    s_inv = np.diag(1. / s)
    s = np.diag(s)
    whitening = np.dot(np.dot(U, s_inv), U.T)
    dewhitening = np.dot(np.dot(U, s), U.T)

    return np.asarray(np.dot(X, whitening), dtype=np.float32)


if __name__ == '__main__':
    train_set, test_set = load_cifar_10()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set

    img_shape = (3, 32, 32)
    # print train_set_x.get_value(borrow = True).shape
    #temp_x = train_set_x.get_value(borrow = True)
    print train_set_x.shape

    for i in range(100):
        img = train_set_x[i].reshape(img_shape).transpose(1, 2, 0)
        pylab.subplot(10, 10, i + 1)
        pylab.axis('off')
        pylab.imshow(img)

    pylab.show()

    #temp_x = np.reshape(test_set_x, (-1, 3, 32, 32))
    #temp_x = np.transpose(temp_x, (0, 3, 2, 1))

    #patches = np.zeros((50000*27*27, 108), dtype = 'float32')

    #patch_width = 6
    # patches = extract_patches_for_classification(dataset = temp_x,
    #                                             patch_width = patch_width
    #                                             )
    # for i in range(2):
    #    patch = extract_patches_for_classification(dataset = temp_x[5000*i:5000*(i+1)],
    #                                               patch_width = patch_width
    #                                               )
    #    patches[5000*27*27*i:5000*27*27*(i+1)] = patch
    # patches1 = extract_patches_for_classification(dataset = temp_x[0:5000],
    #                                             patch_width = patch_width
    #                                             )
    #patches[0:5000*27*27] = patches1

    # patches2 = extract_patches_for_classification(dataset = temp_x[5000:10000],
    #                                             patch_width = patch_width
    #                                             )
    #patches[5000*27*27:5000*27*27*2] = patches2
    #patches = np.concatenate((patches1, patches2))
    # patches.tofile('test_patches.bin')
    # print patches.shape
