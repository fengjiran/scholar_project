from __future__ import division

import os
import sys
import time
import timeit
import cPickle

import numpy as np
import pylab as pl

import theano
import theano.tensor as T
from theano.tensor.signal import pool

import load_cifar10


def Relu(x):
    return T.maximum(0, x)


# step function, the gradient of relu
def step_func(x):
    return x >= 0


# Encoder
def encoder_relu(x):
    return np.maximum(0, x)


class HiddenLayer(object):

    def __init__(self,
                 rng,
                 input,
                 n_in,
                 n_out,
                 W=None,
                 b=None,
                 activation=Relu):
        '''
        Typical hidden layer of a MLP: units are fully-connected and have sigmoidal activation 
        function. Weight maxtrix W is of shape (n_in, n_out) and the bias vector b is of shape
        (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non-linearity to be applied in the hidden layer
        '''
        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(
                rng.normal(
                    loc=0,
                    scale=0.001,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.ones((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        # if W is None:
        #     W_values = np.asarray(
        #         rng.uniform(
        #             low=-np.sqrt(6. / (n_in + n_out)),
        #             high=np.sqrt(6. / (n_in + n_out)),
        #             size=(n_in, n_out)
        #         ),
        #         dtype=theano.config.floatX
        #     )
        #     if activation == T.nnet.sigmoid:
        #         W_values *= 4

        #     W = theano.shared(value=W_values, name='W', borrow=True)

        # if b is None:
        #     b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        #     b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        self.params = [self.W, self.b]

        lin_output = T.dot(input, self.W) + self.b

        self.lin_output = lin_output

        self.output = (lin_output if activation is None
                       else activation(lin_output)
                       )

        lin_output_mirror = -T.dot(input, self.W) + self.b
        self.output_mirror = (lin_output_mirror if activation is None
                              else activation(lin_output_mirror)
                              )

    def competitive_epls_scan(self, inhibitor, pretrain_mini_batch, n_samples):
        self.inhibitor = inhibitor
        l2input = T.sqrt(T.sum(self.input ** 2, axis=1))
        input_ = self.input / l2input[:, np.newaxis]

        l2W = T.sqrt(T.sum(self.W ** 2, axis=0))
        W_ = self.W / l2W

        cosine_sim = T.dot(input_, W_)
        # cosine_sim = self.output

        self.H = (cosine_sim - T.min(cosine_sim)) / (T.max(cosine_sim) - T.min(cosine_sim))

        def compute_abs_competitive_matrix(index, outpt, inpt, W):
            '''
            ||x-w||
            '''
            W = W.T
            x_i = inpt[index]
            outpt = T.set_subtensor(outpt[index], T.abs_(T.sum(x_i - W, axis=1)))
            return outpt

        results1, updates1 = theano.scan(fn=compute_abs_competitive_matrix,
                                         sequences=T.arange(pretrain_mini_batch),
                                         outputs_info=T.zeros_like(self.H),
                                         non_sequences=[input_, W_]
                                         )

        # self.H = (results1[-1] - T.min(results1[-1])) / (T.max(results1[-1] - T.min(results1[-1])))

        def set_value_at_position(index, target, H):
            h = H[index]
            t = target[index]
            k = T.argmax(h - self.inhibitor)

            target = T.set_subtensor(target[index], T.set_subtensor(t[k], 1.0))
            self.inhibitor = T.set_subtensor(self.inhibitor[k], self.inhibitor[k] + H.shape[1] / n_samples)
            return target

        results, updates = theano.scan(fn=set_value_at_position,
                                       sequences=T.arange(pretrain_mini_batch),
                                       outputs_info=T.zeros_like(self.H),
                                       non_sequences=self.H
                                       )

        return 0.5 * T.sum((self.output - results[-1]) ** 2) / pretrain_mini_batch

    def competitive_epls_scan_k(self, inhibitor, pretrain_mini_batch, n_samples, sparsity_rate):
        self.inhibitor = inhibitor
        l2input = T.sqrt(T.sum(self.input ** 2, axis=1))
        input_ = self.input / l2input[:, np.newaxis]

        l2W = T.sqrt(T.sum(self.W ** 2, axis=0))
        W_ = self.W / l2W

        cosine_sim = T.dot(input_, W_)
        # cosine_sim = self.output
        self.H = (cosine_sim - T.min(cosine_sim)) / (T.max(cosine_sim) - T.min(cosine_sim))

        def set_value_at_position(index, target, H):
            h = H[index]
            t = target[index]
            indices = T.argsort(h - self.inhibitor)
            k = T.floor((1 - sparsity_rate) * H.shape[1]).astype('int32')

            target = T.set_subtensor(target[index], T.set_subtensor(t[indices[-k:]], 1.0))

            delta = 1 / ((1 - sparsity_rate) * n_samples)
            self.inhibitor = T.set_subtensor(self.inhibitor[indices[-k:]], self.inhibitor[indices[-k:]] + delta)

            return target

        results, updates = theano.scan(fn=set_value_at_position,
                                       sequences=T.arange(pretrain_mini_batch),
                                       outputs_info=T.zeros_like(self.H),
                                       non_sequences=self.H
                                       )

        self.target = results[-1]

        return 0.5 * T.sum((self.output - results[-1]) ** 2) / pretrain_mini_batch


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
        # initialize the weights W as a matrix of shape (n_in, n_out)
        rng = np.random.RandomState(1234)
        if W is None:
            self.W = theano.shared(
                value=np.asarray(
                    rng.normal(
                        loc=0,
                        scale=0.01,
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = W
        # if W is None:
        #     self.W = theano.shared(
        #         value=np.zeros(
        #             (n_in, n_out),
        #             dtype=theano.config.floatX
        #         ),
        #         name='W',
        #         borrow=True
        #     )
        # else:
        #     self.W = W

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

        # initialize the weights W as a matrix of shape (n_in, n_out)

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

        # rng = np.random.RandomState(1234)
        # if W is None:
        #     self.W = theano.shared(
        #         value=np.asarray(
        #             rng.normal(
        #                 loc=0,
        #                 scale=0.01,
        #                 size=(n_in, n_out)
        #             ),
        #             dtype=theano.config.floatX
        #         ),
        #         name='W',
        #         borrow=True
        #     )
        # else:
        #     self.W = W

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


class Network_epls(object):

    def __init__(self,
                 rng,
                 n_in,
                 hidden_layer_size,
                 n_out,
                 pretrain_mini_batch,
                 mini_batch,
                 n_samples,
                 activation,
                 classifier='LR',
                 sparsity_rate=0.5,
                 params=None
                 ):
        '''
        classifier: LR or SVM
        '''
        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        # self.classifier_x = T.matrix('classifier_x')
        # self.classifier_y = T.ivector('classifier_y')

        self.inhibitor = T.dvector('inhibitor')
        self.l2_reg = T.dscalar('l2_reg')

        self.hiddenLayer = HiddenLayer(rng=rng,
                                       input=self.x,
                                       n_in=n_in,
                                       n_out=hidden_layer_size,
                                       W=params[0] if params else None,
                                       b=params[1] if params else None,
                                       activation=activation
                                       )

        self.pretrain_cost = self.hiddenLayer.competitive_epls_scan_k(self.inhibitor,
                                                                      pretrain_mini_batch,
                                                                      n_samples,
                                                                      sparsity_rate
                                                                      )

        self.pretrain_params = self.hiddenLayer.params

        classifier_input = T.transpose(self.hiddenLayer.output, (1, 0))  # (1600, N*14*14)
        classifier_input1 = T.reshape(classifier_input, (hidden_layer_size, mini_batch, 14, 14))   # (1600, N, 14, 14)
        classifier_input2 = pool.pool_2d(input=classifier_input1,
                                         ds=(7, 7),
                                         ignore_border=True
                                         )  # (1600, N, 2, 2)
        classifier_input3 = T.transpose(classifier_input2, (1, 0, 2, 3))  # (N, 1600, 2, 2)
        classifier_input4 = T.reshape(classifier_input3, (mini_batch, 4 * hidden_layer_size))  # (N, 4*1600)

        # polarity splitting
        classifier_input = T.transpose(self.hiddenLayer.output_mirror, (1, 0))  # (1600, N*14*14)
        classifier_input1 = T.reshape(classifier_input, (hidden_layer_size, mini_batch, 14, 14))   # (1600, N, 14, 14)
        classifier_input2 = pool.pool_2d(input=classifier_input1,
                                         ds=(7, 7),
                                         ignore_border=True
                                         )  # (1600, N, 2, 2)
        classifier_input3 = T.transpose(classifier_input2, (1, 0, 2, 3))  # (N, 1600, 2, 2)
        classifier_input4_mirror = T.reshape(classifier_input3, (mini_batch, 4 * hidden_layer_size))  # (N, 4*1600)

        classifier_input = T.concatenate([classifier_input4, classifier_input4_mirror], axis=1)

        if classifier == 'LR':
            self.classifier = LogisticRegression(input=classifier_input,
                                                 n_in=4 * 2 * hidden_layer_size,
                                                 n_out=n_out,
                                                 W=params[-2] if params else None,
                                                 b=params[-1] if params else None
                                                 )

            # the cost and parameters of EPLS layer
            # self.pretrain_cost = self.hiddenLayer.competitive_epls_scan(self.inhibitor,
            #                                                             mini_batch,
            #                                                             n_samples
            #                                                             )

            self.pretrain_cost = self.hiddenLayer.competitive_epls_scan_k(self.inhibitor,
                                                                          pretrain_mini_batch,
                                                                          n_samples,
                                                                          sparsity_rate
                                                                          )
            self.pretrain_params = self.hiddenLayer.params

            # the cost and parameters of classifier layer
            self.classifier_cost = self.classifier.negative_log_likelihood(self.y) + self.l2_reg * ((self.classifier.W ** 2).sum() + (self.hiddenLayer.W ** 2).sum())
            self.classifier_params = self.classifier.params
            self.errors = self.classifier.errors(self.y)

            # the parameters of the network
            self.params = []
            self.params.extend(self.pretrain_params)
            self.params.extend(self.classifier_params)

        else:
            self.classifier = OVASVMLayer(input=classifier_input,
                                          n_in=4 * 2 * hidden_layer_size,
                                          n_out=n_out,
                                          W=params[-2] if params else None,
                                          b=params[-1] if params else None
                                          )

            # the cost and parameters of EPLS layer
            # self.pretrain_cost = self.hiddenLayer.competitive_epls_scan(self.inhibitor,
            #                                                             mini_batch,
            #                                                             n_samples
            #                                                             )

            self.pretrain_cost = self.hiddenLayer.competitive_epls_scan_k(self.inhibitor,
                                                                          pretrain_mini_batch,
                                                                          n_samples,
                                                                          sparsity_rate
                                                                          )

            self.pretrain_params = self.hiddenLayer.params

            # the cost and parameters of classifier layer
            self.classifier_cost = self.classifier.ova_svm_cost(self.y) + self.l2_reg * ((self.classifier.W ** 2).sum() + (self.hiddenLayer.W ** 2).sum())
            self.classifier_params = self.classifier.params
            self.errors = self.classifier.errors(self.y)

            # the parameters of the network
            self.params = []
            self.params.extend(self.pretrain_params)
            self.params.extend(self.classifier_params)

    def pretraining_function(self, train_set_x, pretrain_batch_size, learning_rate):
        """
        Generates a function implementing training the EPLS layer.
        """
        # index to a mini-batch
        index = T.lscalar('index')

        gradients = T.dmatrix('gradients')   # for testing the gradients

        # compute the gradient of cost with respect to the theta (stored in params)
        # the resulting gradients will be stored in a list gparams

        gparams = [T.grad(self.pretrain_cost, param)
                   for param in self.pretrain_params]

        # gparam_w = T.dot(self.x.T, (self.hiddenLayer.output - self.hiddenLayer.target) * step_func(self.hiddenLayer.lin_output)).astype('float32') / pretrain_batch_size
        # gparam_b = T.dot(step_func(self.hiddenLayer.lin_output).sum(axis=1).T, self.hiddenLayer.output - self.hiddenLayer.target).astype('float32') / pretrain_batch_size

        # gparam_w = T.dot(self.x.T, (self.hiddenLayer.output - self.hiddenLayer.target) * self.hiddenLayer.output * (1 - self.hiddenLayer.output)).astype('float32') / pretrain_batch_size
        # gparam_b = T.dot((self.hiddenLayer.output * (1 - self.hiddenLayer.output)).sum(axis=1).T, self.hiddenLayer.output - self.hiddenLayer.target).astype('float32') / pretrain_batch_size

        # gparams = [gparam_w, gparam_b]

        updates = [(param, param - learning_rate * gparam)
                   for param, gparam in zip(self.pretrain_params, gparams)
                   ]

        gradients = gparams[0]

        pretrain_fn = theano.function(
            inputs=[index, self.inhibitor],
            outputs=[self.pretrain_cost, self.hiddenLayer.output, gradients],
            updates=updates,
            givens={
                self.x: train_set_x[index * pretrain_batch_size: (index + 1) * pretrain_batch_size]
            }
        )

        return pretrain_fn

    def build_finetune_function(self, datasets, batch_size, learning_rate):
        """
        Generates a function 'train' that implementa one step of finetuning,
        a function 'validate' that computes the error on a batch from the 
        validation set, and a function 'test' that computes the error on a 
        batch from the test set.
        """
        (train_set_x, train_set_y) = datasets[0]
        (test_set_x, test_set_y) = datasets[1]

        # compute number of minibatchs for training and testing
        n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
        n_test_batches = int(test_set_x.get_value(borrow=True).shape[0] / batch_size)

        # index to a mini-batch
        index = T.lscalar('index')
        # learning_rate = T.dscalar('learning_rate')

        gradients1, gradients2 = T.dmatrices('gradients1', 'gradients2')   # for testing the gradients

        gparams = [T.grad(self.classifier_cost, param)
                   for param in self.params]

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        gradients1 = gparams[0]
        gradients2 = gparams[2]

        train_fn = theano.function(
            inputs=[index, self.l2_reg],
            outputs=[self.classifier_cost, gradients1, gradients2],
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
                self.x: test_set_x[index * batch_size * 196: (index + 1) * batch_size * 196],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
            },
            name='test'
        )

        # Create a function that scans the entire test set
        def test_score(n_batches):
            return [test_score_i(i) for i in xrange(n_batches)]

        return train_fn, test_score

    def train_classifier(self, datasets, batch_size, learning_rate):
        '''
        '''
        self.classifier_x = T.matrix('classifier_x')
        self.classifier_y = T.ivector('classifier_y')

        (train_set_x, train_set_y) = datasets[0]
        (test_set_x, test_set_y) = datasets[1]


def training_function(pretrain_lr=0.13,
                      finetune_lr=[],
                      n_in=6 * 6 * 3,
                      hidden_layer_size=1600,
                      n_out=10,
                      pretraining_epochs=20,
                      training_epochs=1000,
                      pretrain_batch_size=1600,
                      batch_size=100,
                      eps=1e-6,
                      classifier='LR',
                      l2_reg=0.,
                      activation=Relu,
                      sparsity_rate=0.5,
                      continue_train=False
                      ):
    '''
    '''
    rng = np.random.RandomState(1234)
    patch_width = 6

    print '... loading the data'

    [(train_set_x, train_set_y), (test_set_x, test_set_y)] = load_cifar10.load_cifar_10()

    temp_x = np.reshape(train_set_x, (-1, 3, 32, 32))
    temp_x = np.transpose(temp_x, (0, 3, 2, 1))

    pretrain_patches = load_cifar10.extract_patches_for_pretrain(dataset=temp_x,
                                                                 NSPL=400000,
                                                                 patch_width=patch_width
                                                                 )
    # local contrast normalize
    data_mean = pretrain_patches.mean(axis=0)
    pretrain_patches -= data_mean

    normalizers = np.sqrt(10 + pretrain_patches.var(axis=0, ddof=1))
    normalizers[normalizers < 1e-8] = 1.
    # pretrain_patches /= normalizers

    # pretrain_patches = load_cifar10.global_contrast_normalize(pretrain_patches)
    # pretrain_patches = load_cifar10.zca(pretrain_patches)
    pretrain_patches = np.asarray(pretrain_patches, dtype=np.float32)

    n_samples = pretrain_patches.shape[0]

    shared_pretrain_patches = load_cifar10.shared_dataset_x(pretrain_patches)

    # print pretrain_patches.shape
    print '... building the model'

    # construct the network
    if continue_train:
        params = []
        save_file = open('params.save')
        pas = cPickle.load(save_file)
        for pa in pas:
            params.append(theano.shared(
                np.asarray(pa, dtype=theano.config.floatX)))

        save_file.close()

    net = Network_epls(rng=rng,
                       n_in=n_in,
                       hidden_layer_size=hidden_layer_size,
                       n_out=n_out,
                       pretrain_mini_batch=pretrain_batch_size,
                       mini_batch=batch_size,
                       n_samples=n_samples,
                       sparsity_rate=sparsity_rate,
                       activation=activation
                       )

    n_train_batches = int(n_samples / pretrain_batch_size)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'

    pretraining_fn = net.pretraining_function(
        train_set_x=shared_pretrain_patches,
        pretrain_batch_size=pretrain_batch_size,
        learning_rate=pretrain_lr
    )

    print '... pre-training the model'
    start_time = time.clock()

    done_looping = False
    epoch = 0

    epoch_error = []
    # minibatch_output = np.zeros((batch_size, hidden_layer_size))

    while (epoch < pretraining_epochs) and (not done_looping):

        arr = range(n_train_batches)
        np.random.shuffle(arr)
        inhibitor = np.zeros((hidden_layer_size,))  # the inhibitor

        minibatch_error = []
        for minibatch_index in arr:
            minibatch_avg_cost, minibatch_output, gradients = pretraining_fn(minibatch_index, inhibitor)
            minibatch_error.append(minibatch_avg_cost)
            # print 'Batch {0} gradient abs sum is {1}'.format(minibatch_index, abs(np.asarray(gradients)).sum())
            print abs(np.asarray(gradients)).sum()

        loss = np.mean(minibatch_error)
        print 'The loss of epoch {0} is {1}'.format(epoch, loss)

        epoch_error.append(loss)

        # stop condition
        # The relative decrement error between epochs is smaller than eps
        if epoch > 0:
            err = (epoch_error[-2] - epoch_error[-1]) / epoch_error[-2]
            if err < eps:
                done_looping = True

        epoch = epoch + 1

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # np.savetxt('hiddenLayer_output.txt', minibatch_output, fmt='%f', delimiter=',')

    # save the params of pretraining
    save_file = open('pretrain_params.save', 'wb')
    temp = []
    for param in net.pretrain_params:
        temp.append(param.get_value(borrow=True))
    cPickle.dump(temp, save_file, True)
    save_file.close()

    ###########################
    # TRAINING THE CLASSIFIER #
    ###########################
    print '... training the classifier'

    ntrain = train_set_y.shape[0]
    ntest = test_set_y.shape[0]

    # load the parameters of first layer
    files = open('pretrain_params.save')
    pas = cPickle.load(files)
    w = pas[0]
    b = pas[1]

    print '... Preparing the data'

    temp_x = np.reshape(train_set_x, (-1, 3, 32, 32))
    temp_x = np.transpose(temp_x, (0, 3, 2, 1))
    train_patches = load_cifar10.extract_patches_for_classification(dataset=temp_x,
                                                                    n_patches_per_image=14 * 14,
                                                                    patch_width=patch_width
                                                                    )
    # local contrast normalize
    train_data_mean = train_patches.mean(axis=0)
    train_patches -= train_data_mean
    normalizers = np.sqrt(10 + train_patches.var(axis=0, ddof=1))
    normalizers[normalizers < 1e-8] = 1.
    # train_patches /= normalizers

    train_set_x = encoder_relu(np.dot(train_patches, w) + b)
    train_set_x = np.transpose(train_set_x, (1, 0))
    train_set_x = np.reshape(train_set_x, (hidden_layer_size, batch_size, 14, 14))

    # shared_train_x = load_cifar10.shared_dataset_x(train_patches)
    # shared_train_y = load_cifar10.shared_dataset_y(train_set_y)

    temp_x = np.reshape(test_set_x, (-1, 3, 32, 32))
    temp_x = np.transpose(temp_x, (0, 3, 2, 1))
    test_patches = load_cifar10.extract_patches_for_classification(dataset=temp_x,
                                                                   n_patches_per_image=14 * 14,
                                                                   patch_width=patch_width
                                                                   )

    test_patches -= train_data_mean
    # test_patches /= normalizers
    shared_test_x = load_cifar10.shared_dataset_x(test_patches)
    shared_test_y = load_cifar10.shared_dataset_y(test_set_y)

    train_set = (shared_train_x, shared_train_y)
    test_set = (shared_test_x, shared_test_y)
    datasets = (train_set, test_set)

    n_train_batches = int(ntrain / batch_size)
    n_test_batches = int(ntest / batch_size)

    start_time = timeit.default_timer()

    for i in range(len(finetune_lr)):
        print '...... for finetune_learning_rate_{0}: {1}'.format(i, finetune_lr[i])
        print '... getting the finetuning functions'
        train_fn, test_model = net.build_finetune_function(datasets=datasets,
                                                           batch_size=batch_size,
                                                           learning_rate=finetune_lr[i]
                                                           )

        best_validation_loss = np.inf
        epoch = 0
        while (epoch < training_epochs):
            print 'Training epoch {0}'.format(epoch)
            epoch += 1
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost, dw1, dw2 = train_fn(minibatch_index, l2_reg)
                iter = (epoch - 1) * n_train_batches + minibatch_index
                # print 'Batch {0}: dw1={1}, dw2={2}'.format(minibatch_index, abs(np.asarray(dw1)).sum(), abs(np.asarray(dw2)).sum())

            print 'Testing...'
            validation_losses = test_model(n_test_batches)
            this_validation_loss = np.mean(validation_losses)
            print 'Epoch {0}, Test error {1}%'.format(epoch - 1, this_validation_loss * 100)

            if this_validation_loss < best_validation_loss:
                best_validation_loss = this_validation_loss

    end_time = timeit.default_timer()

    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':

    l2_reg = 0.0005  # weight of weight decay
    finetune_lr = [0.01, 0.002, 0.0004, 8e-5, 1.6e-5]

    training_function(pretrain_lr=0.005,  # 0.15
                      finetune_lr=finetune_lr,  # 0.05
                      n_in=6 * 6 * 3,
                      hidden_layer_size=1600,
                      n_out=10,
                      pretraining_epochs=10,
                      training_epochs=150,
                      pretrain_batch_size=1600,
                      batch_size=100,
                      eps=1e-6,
                      classifier='LR',
                      l2_reg=l2_reg,
                      activation=Relu,
                      sparsity_rate=0.8,
                      continue_train=False
                      )

    print 'DONE!'
