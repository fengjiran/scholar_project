"""
Sparse Autoencoder implementtation by theano
"""
import os
import sys
import time
import timeit

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import matplotlib.pyplot

import load_cifar10


def Relu(x):
    return T.maximum(0, x)


class SparseAutoencoder(object):
    """Sparse Autoencoder class (SA)
    Initialization of autoencoder object
    """

    def __init__(self,
                 numpy_rng,
                 n_visible,
                 n_hidden,
                 inpt,
                 theano_rng=None,
                 W=None,
                 bhid=None,
                 bvis=None,
                 activation=T.nnet.sigmoid
                 ):
        '''
        Initialize parameters of the sparse autoencoder object
        '''
        #self.x = T.matrix('x')
        self.inpt = inpt
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.activation = activation

        # desired average activation of hidden units
        # self.rho = T.dscalar('rho')
        # self.lamda = T.dscalar('lamda')  # weight decay parameter
        # self.beta = T.dscalar('beta')  # weight of sparity penalty term

        # create a theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng

        self.params = [self.W, self.b, self.b_prime]

        self.hidden_values = self.activation(T.dot(self.inpt, self.W) + self.b)
        self.reconstructed_input = self.activation(T.dot(self.hidden_values, self.W_prime) + self.b_prime)

    def get_hidden_values(self, inpt):
        """Computes the values of the hidden layer"""
        return self.activation(T.dot(inpt, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the value of
           the hidden layer
        """
        return self.activation(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost(self, rho, lamda, beta):
        """
        This function computes the cost and the updates for one training
        step of the SA.
        rho: desired average activation of hidden units
        lamda: weight decay parameter
        beta: weight of sparity penalty term
        """
        hidden_layer = self.get_hidden_values(self.inpt)
        rho_cap = T.mean(hidden_layer, axis=1)

        output_layer = self.get_reconstructed_input(hidden_layer)

        diff = output_layer - self.inpt

        sum_of_square_error = 0.5 * T.sum(T.square(diff)) / self.inpt.shape[0]

        cross_entropy = -T.mean(T.sum(self.inpt * T.log(output_layer) + (1 - self.inpt) * T.log(1 - output_layer), axis=1))

        weight_decay = 0.5 * lamda * (T.sum(T.square(self.W)) + T.sum(T.square(self.W_prime)))

        KL_devergence = beta * T.sum(rho * T.log(rho / rho_cap) + (1 - rho) * T.log((1 - rho) / (1 - rho_cap)))

        cost = sum_of_square_error + weight_decay + KL_devergence

        # gparams = T.grad(cost, self.params)

        # updates = [(param, param - lr * gparam)
        #            for param, gparam in zip(self.params, gparams)]

        return cost


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = W

        # initialize the biases b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b = b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class OVASVMLayer(object):
    """SVM-like layer
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = W

        # self.W = theano.shared(value=np.zeros((n_in, n_out),
        #                                          dtype=theano.config.floatX),
        #                         name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b = b

        # self.b = theano.shared(value=np.zeros((n_out,),
        #                                          dtype=theano.config.floatX),
        #                        name='b', borrow=True)

        # parameters of the model
        self.params = [self.W, self.b]

        self.output = T.dot(input, self.W) + self.b

        self.y_pred = T.argmax(self.output, axis=1)

        self.L2 = (self.W ** 2).sum()

    def hinge(self, u):
        return T.maximum(0, 1 - u)

    def ova_svm_cost(self, y1):
        """ return the one-vs-all svm cost
        given ground-truth y in one-hot {-1, 1} form """
        y1_printed = theano.printing.Print('this is important')(T.max(y1))
        margin = y1 * self.output
        cost = (self.hinge(margin) ** 2).mean(axis=0).sum()
        return cost

    def errors(self, y):
        """ compute zero-one loss
        note, y is in integer form, not one-hot
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class network(object):

    def __init__(self,
                 n_visible,
                 n_hidden,
                 n_out,
                 batch_size,
                 activation,
                 classifier,
                 params=None
                 ):
        '''
        classifier: LR or SVM
        rho: desired average activation of hidden units
        lamda: weight decay parameter
        beta: weight of sparity penalty term
        l2_reg: weight of classifier weight decay parameter
        '''

        rng = np.random.RandomState(1234)

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        self.rho = T.dscalar('rho')
        self.lamda = T.dscalar('lamda')
        self.beta = T.dscalar('beta')
        self.l2_reg = T.scalar('l2_reg')

        self.sparseAutoencoder = SparseAutoencoder(numpy_rng=rng,
                                                   n_visible=n_visible,
                                                   n_hidden=n_hidden,
                                                   inpt=self.x,
                                                   W=params[0] if params else None,
                                                   bhid=params[1] if params else None,
                                                   bvis=params[2] if params else None,
                                                   activation=activation
                                                   )

        classifier_input = T.transpose(self.sparseAutoencoder.hidden_values, (1, 0))
        classifier_input1 = T.reshape(classifier_input, (n_hidden, batch_size, 14, 14))
        classifier_input2 = pool.pool_2d(input=classifier_input1,
                                         ds=(7, 7),
                                         ignore_border=True
                                         )
        classifier_input3 = T.transpose(classifier_input2, (1, 0, 2, 3))
        classifier_input4 = T.reshape(classifier_input3, (batch_size, 4 * n_hidden))

        if classifier == 'LR':
            self.classifier = LogisticRegression(input=classifier_input4,
                                                 n_in=4 * n_hidden,
                                                 n_out=n_out,
                                                 W=params[-2] if params else None,
                                                 b=params[-1] if params else None
                                                 )

            self.pretrain_cost = self.sparseAutoencoder.get_cost(rho=self.rho,
                                                                 lamda=self.lamda,
                                                                 beta=self.beta
                                                                 )

            self.pretrain_params = self.sparseAutoencoder.params

            self.classifier_cost = self.classifier.negative_log_likelihood(self.y) + self.l2_reg * (self.classifier.W ** 2).sum()
            self.classifier_params = self.classifier.params
            self.errors = self.classifier.errors(self.y)

            # the parameters of the network
            self.params = []
            self.params.extend(self.pretrain_params)
            self.params.extend(self.classifier_params)

        else:
            self.classifier = OVASVMLayer(input=classifier_input4,
                                          n_in=4 * n_hidden,
                                          n_out=n_out,
                                          W=params[-2] if params else None,
                                          b=params[-1] if params else None
                                          )

            self.pretrain_cost = self.sparseAutoencoder.get_cost(rho=self.rho,
                                                                 lamda=self.lamda,
                                                                 beta=self.beta
                                                                 )

            self.pretrain_params = self.sparseAutoencoder.params

            self.classifier_cost = self.classifier.ova_svm_cost(self.y) + self.l2_reg * self.classifier.L2
            self.classifier_params = self.classifier.params
            self.errors = self.classifier.errors(self.y)

            # the parameters of the network
            self.params = []
            self.params.extend(self.pretrain_params)
            self.params.extend(self.classifier_params)

    def pretrain_function(self, train_set_x, batch_size, lr):

        index = T.lscalar('index')

        gparams = [
            T.grad(self.pretrain_cost, param)
            for param in self.pretrain_params
        ]

        updates = [
            (param, param - lr * gparam)
            for param, gparam in zip(self.pretrain_params, gparams)
        ]

        pretrain_fn = theano.function(
            inputs=[index, self.rho, self.lamda, self.beta],
            outputs=[self.pretrain_cost, self.sparseAutoencoder.hidden_values],
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

        return pretrain_fn

    def build_finetune_function(self, datasets, batch_size, lr):
        (train_set_x, train_set_y) = datasets[0]
        (test_set_x, test_set_y) = datasets[1]

        # n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
        # n_test_batches = int(test_set_x.get_value(borrow=True).shape[0] / batch_size)

        index = T.lscalar('index')

        gparams = [
            T.grad(self.classifier_cost, param)
            for param in self.params
        ]

        updates = [
            (param, param - lr * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index, self.l2_reg],
            outputs=self.classifier_cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size * 196: (index + 1) * batch_size * 196],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            },
            name='train'
        )

        test_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x: train_set_x[index * batch_size * 196: (index + 1) * batch_size * 196],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            },
            name='test'
        )

        def test_score(n_batches):
            return [test_score_i(i) for i in xrange(n_batches)]

        return train_fn, test_score


def train_function(pretrain_lr=0.13,
                   finetune_lr=0.05,
                   n_in=6 * 6 * 3,
                   n_hidden=1600,
                   n_out=10,
                   pretrain_epochs=50,
                   finetune_epochs=1000,
                   batch_size=100,
                   eps=1e-6,
                   classifier='LR',
                   rho=0.01,
                   lamda=0.0001,
                   beta=3,
                   l2_reg=[],
                   activation=T.nnet.sigmoid,
                   ):
    '''
    '''

    rng = np.random.RandomState(1234)
    patch_width = 6

    print "... loading the data"

    [(train_set_x, train_set_y), (test_set_x, test_set_x)] = load_cifar10.load_cifar_10()

    temp_x = np.reshape(train_set_x, (-1, 3, 32, 32))
    temp_x = np.transpose(temp_x, (0, 3, 2, 1))

    pretrain_patches = load_cifar10.extract_patches_for_pretrain(dataset=temp_x,
                                                                 NSPL=400000,
                                                                 patch_width=patch_width
                                                                 )

    pretrain_patches = load_cifar10.global_contrast_normalize(pretrain_patches)
    pretrain_patches = load_cifar10.zca(pretrain_patches)

    n_samples = pretrain_patches.shape[0]

    shared_pretrain_patches = load_cifar10.shared_dataset_x(pretrain_patches)

    print "... building the model"

    # contruct the network
    net = network(n_visible=n_in,
                  n_hidden=n_hidden,
                  n_out=n_out,
                  batch_size=batch_size,
                  activation=activation,
                  classifier=classifier
                  )

    n_train_batches = int(n_samples / batch_size)

    ##### pretraining the model ####
    print "... getting the pretraining functions"

    pretrain_fn = net.pretrain_function(train_set_x=shared_pretrain_patches,
                                        batch_size=batch_size,
                                        lr=pretrain_lr
                                        )

    print "... pre-training the model"
    start_time = time.clock()
    done_looping = False
    epoch = 0
    epoch_error = []

    while (epoch < pretrain_epochs) and (not done_looping):
        arr = range(n_train_batches)
        np.random.shuffle(arr)

        minibatch_error = []
        for minibatch_index in arr:
            minibatch_cost, minibatch_output = pretrain_fn(minibatch_index,
                                                           rho,
                                                           lamda,
                                                           beta
                                                           )

            minibatch_error.append(minibatch_cost)

        loss = np.mean(minibatch_error)
        print "The loss of epoch {0} is {1}".format(epoch, loss)

        epoch_error.append(loss)

        # stop condition
        # The relative decrement error between epochs is smaller than eps
        if epoch > 0:
            err = (epoch_error[-2] - epoch_error[-1]) / epoch_error[-2]
            if err < eps:
                done_looping = True

        epoch = epoch + 1

    end_time = time.clock()

    print >> sys.stderr, ('The pretrain code for file' +
                          os.path.split(__file__)[1] +
                          'ran for %.2f m' % ((end_time - start_time) / 60.))


if __name__ == '__main__':

    train_function(pretrain_lr=0.25,
                   finetune_lr=0.05,
                   n_in=6 * 6 * 3,
                   n_hidden=1600,
                   n_out=10,
                   pretrain_epochs=50,
                   finetune_epochs=1000,
                   batch_size=100,
                   eps=1e-6,
                   classifier='LR',
                   rho=0.01,
                   lamda=0.0001,
                   beta=3,
                   l2_reg=[],
                   activation=T.nnet.sigmoid,
                   )
