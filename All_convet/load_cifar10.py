"""
http://www.cs.toronto.edu/~kriz/cifar.html.

https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/datasets/cifar10.py.

"""

import os
import cPickle
import numpy as np
# import scipy
from scipy import linalg
from keras.utils.np_utils import to_categorical


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
        a /= 255.0
        train_set_x[i * 10000:(i + 1) * 10000, :] = a
        train_set_y[i * 10000:(i + 1) * 10000] = dicts['labels']
        fo.close()

    fo = open(os.path.join(datapath, 'test_batch'), 'rb')
    dicts = cPickle.load(fo)
    a = np.copy(dicts['data']).astype('float32')
    a /= 255.0

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


def global_contrast_normalize(X, scale=1.0, subtract_mean=True, use_std=True, sqrt_bias=10.0, min_divisor=1e-8):
    """GCN.

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
        # X -= mean[:, np.newaxis]
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
    U, S, _ = linalg.svd(cov)
    s = np.sqrt(S.clip(epsilon))
    s_inv = np.diag(1. / s)
    s = np.diag(s)
    whitening = np.dot(np.dot(U, s_inv), U.T)
    # dewhitening = np.dot(np.dot(U, s), U.T)

    return np.asarray(np.dot(X, whitening), dtype=np.float32)


if __name__ == '__main__':
    train_set, test_set = load_cifar_10()
    train_set_x, train_set_y = train_set
    test_set_x, test_set_y = test_set

    img_shape = (3, 32, 32)
    # print train_set_x.get_value(borrow = True).shape
    # temp_x = train_set_x.get_value(borrow = True)
    print train_set_x.shape

    # for i in range(100):
    #     img = train_set_x[i].reshape(img_shape).transpose(1, 2, 0)
    #     pylab.subplot(10, 10, i + 1)
    #     pylab.axis('off')
    #     pylab.imshow(img)

    # pylab.show()
