from __future__ import division

import os
import sys
import time
import timeit
import cPickle
import csv

import yaml

import numpy as np
import pylab as pl

import theano
import theano.tensor as T
from theano.tensor.signal import pool

import load_cifar10


def Relu(x, alpha=0):
    # return T.maximum(0, x)
    return T.nnet.relu(x, alpha=alpha)


# step function, the gradient of relu
def step_func(x):
    return x >= 0


def to_categorical(y, nb_classes=None):
    '''
    convert class vector (intergers from 0 to nb_classes-1) to binary class matrix.

    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes

    # Returns
        A binary matrix representation of the input.
    '''
    y = np.array(y, dtype='int')

    if not nb_classes:
        nb_classes = np.max(y) + 1

    Y = -1 * np.ones((len(y), nb_classes))

    for i in range(len(y)):
        Y[i, y[i]] = 1

    return Y


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
                    scale=0.01,
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
        # self.target = 2 * self.target - 1  # remapping in [-1, 1]

        return 0.5 * T.sum((self.output - self.target) ** 2) / pretrain_mini_batch
        # return 0.5 * T.sum((self.lin_output - self.target) ** 2) / pretrain_mini_batch


class DropoutLayer(object):

    seed_common = np.random.RandomState(0)
    layers = []

    def __init__(self, input, n_in, n_out, prob_drop=0.2):

        self.prob_drop = prob_drop
        self.prob_keep = 1.0 - prob_drop
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
        self.flag_off = 1.0 - self.flag_on

        seed_this = DropoutLayer.seed_common.randint(0, 2**31 - 1)
        mask_rng = theano.tensor.shared_randomstreams.RandomStreams(seed_this)
        self.mask = mask_rng.binomial(n=1, p=self.prob_keep, size=input.shape)

        self.output = self.flag_on * T.cast(self.mask, theano.config.floatX) * input + self.flag_off * self.prob_keep * input

        DropoutLayer.layers.append(self)

        print 'Dropout layer with P_drop: ' + str(self.prob_drop)

    @staticmethod
    def SetDropoutOn():
        for i in range(len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(1.0)

    @staticmethod
    def SetDropoutOff():
        for i in range(len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(0.0)


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, name='LR', W=None, b=None):
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
                        scale=0.1,
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
        #         value=np.asarray(
        #             rng.uniform(
        #                 low=-np.sqrt(6. / (n_in + n_out)),
        #                 high=np.sqrt(6. / (n_in + n_out)),
        #                 size=(n_in, n_out)
        #             ),
        #             dtype=theano.config.floatX
        #         ),
        #         name='W',
        #         borrow=True
        #     )
        # else:
        #     self.W = W

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

        self.name = name

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

    def __init__(self, input, n_in, n_out, name='svm', W=None, b=None):
        """ Initialize the parameters of the svm

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

        rng = np.random.RandomState(1234)
        if W is None:
            self.W = theano.shared(
                value=np.asarray(
                    rng.normal(
                        loc=0,
                        scale=0.1,
                        size=(n_in, n_out)
                    ),
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

        self.name = name

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

    # def errors(self, y1):

    #     a = y1[T.arange(self.y_pred.shape[0]), self.y_pred]

    #     return 1 - T.mean((a + 1) / 2)


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
                 params=None,
                 polarity_split=False,
                 use_dropout=True,
                 dropout_rate=0.2,
                 max_col_norm=3
                 ):
        '''
        classifier: LR or SVM
        '''
        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.y1 = T.imatrix('y1')  # one-hot encoded labels as {-1, 1}, for svm

        self.inhibitor = T.dvector('inhibitor')
        self.l2_reg = T.dscalar('l2_reg')
        self.l1_reg = T.dscalar('l1_reg')

        self.use_dropout = use_dropout
        self.max_col_norm = max_col_norm

        self.hiddenLayer = HiddenLayer(rng=rng,
                                       input=self.x,
                                       n_in=n_in,
                                       n_out=hidden_layer_size,
                                       W=params[0] if params else None,
                                       b=params[1] if params else None,
                                       activation=activation
                                       )

        # the cost and parameters of EPLS layer
        self.pretrain_cost = self.hiddenLayer.competitive_epls_scan_k(self.inhibitor,
                                                                      pretrain_mini_batch,
                                                                      n_samples,
                                                                      sparsity_rate
                                                                      )

        self.pretrain_params = self.hiddenLayer.params

        # polarity splitting
        if polarity_split:

            classifier_input = T.concatenate([self.hiddenLayer.output, self.hiddenLayer.output_mirror], axis=1)  # (N*27*27, 1600*2)

            classifier_input = T.transpose(classifier_input, (1, 0))  # (1600*2, N*27*27)

            classifier_input = T.reshape(classifier_input, (hidden_layer_size * 2, mini_batch, 27, 27))   # (1600*2, N, 27, 27)

            classifier_input = pool.pool_2d(input=classifier_input,
                                            ds=(13, 13),
                                            ignore_border=True
                                            )  # (1600*2, N, 2, 2)

            classifier_input = T.transpose(classifier_input, (1, 0, 2, 3))  # (N, 1600*2, 2, 2)

            classifier_input = T.reshape(classifier_input, (mini_batch, 4 * 2 * hidden_layer_size))  # (N, 8*1600)

            hidden_layer_size = 4 * 2 * hidden_layer_size
        else:
            #
            classifier_input = T.transpose(self.hiddenLayer.output, (1, 0))

            classifier_input = T.reshape(classifier_input, (hidden_layer_size, mini_batch, 27, 27))

            classifier_input = pool.pool_2d(input=classifier_input,
                                            ds=(13, 13),
                                            ignore_border=True
                                            )

            classifier_input = T.transpose(classifier_input, (1, 0, 2, 3))

            classifier_input = T.reshape(classifier_input, (mini_batch, 4 * hidden_layer_size))  # (N, 8*1600)

            hidden_layer_size = 4 * hidden_layer_size

        if use_dropout:

            self.dropout = DropoutLayer(input=classifier_input,
                                        n_in=hidden_layer_size,
                                        n_out=hidden_layer_size,
                                        prob_drop=dropout_rate
                                        )

            classifier_input = self.dropout.output

        if classifier == 'LR':
            self.classifier = LogisticRegression(input=classifier_input,
                                                 n_in=hidden_layer_size,
                                                 n_out=n_out,
                                                 W=params[2] if params else None,
                                                 b=params[3] if params else None
                                                 )

            # the cost and parameters of EPLS layer
            # self.pretrain_cost = self.hiddenLayer.competitive_epls_scan(self.inhibitor,
            #                                                             mini_batch,
            #                                                             n_samples
            #                                                             )

            # self.pretrain_cost = self.hiddenLayer.competitive_epls_scan_k(self.inhibitor,
            #                                                               pretrain_mini_batch,
            #                                                               n_samples,
            #                                                               sparsity_rate
            #                                                               )
            # self.pretrain_params = self.hiddenLayer.params

            # the cost and parameters of classifier layer
            l1 = (abs(self.hiddenLayer.W).sum() + abs(self.classifier.W).sum()) / mini_batch

            l2_sqr = ((self.classifier.W ** 2).sum() + (self.hiddenLayer.W ** 2).sum()) / (2 * mini_batch)

            self.classifier_cost = self.classifier.negative_log_likelihood(self.y) + self.l2_reg * l2_sqr + self.l1_reg * l1
            # self.classifier_cost = self.classifier.negative_log_likelihood(self.y) + self.l2_reg * (self.classifier.W ** 2).sum()
            self.classifier_params = self.classifier.params
            self.errors = self.classifier.errors(self.y)

            # the parameters of the network
            self.params = []
            self.params.extend(self.pretrain_params)
            self.params.extend(self.classifier_params)

        else:
            self.classifier = OVASVMLayer(input=classifier_input,
                                          n_in=hidden_layer_size,
                                          n_out=n_out,
                                          W=params[2] if params else None,
                                          b=params[3] if params else None
                                          )

            # the cost and parameters of EPLS layer
            # self.pretrain_cost = self.hiddenLayer.competitive_epls_scan(self.inhibitor,
            #                                                             mini_batch,
            #                                                             n_samples
            #                                                             )

            # self.pretrain_cost = self.hiddenLayer.competitive_epls_scan_k(self.inhibitor,
            #                                                               pretrain_mini_batch,
            #                                                               n_samples,
            #                                                               sparsity_rate
            #                                                               )

            # self.pretrain_params = self.hiddenLayer.params

            # the cost and parameters of classifier layer
            l1 = (abs(self.hiddenLayer.W).sum() + abs(self.classifier.W).sum()) / mini_batch

            l2_sqr = ((self.classifier.W ** 2).sum() + (self.hiddenLayer.W ** 2).sum()) / (2 * mini_batch)

            self.classifier_cost = self.classifier.ova_svm_cost(self.y1) + self.l2_reg * l2_sqr + self.l1_reg * l1
            # self.classifier_cost = self.classifier.ova_svm_cost(self.y) + self.l2_reg * (self.classifier.W ** 2).sum()
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
        (train_set_x, train_set_y) = datasets
        # (test_set_x, test_set_y) = datasets[1]

        # compute number of minibatchs for training and testing
        # n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
        # n_test_batches = int(test_set_x.get_value(borrow=True).shape[0] / batch_size)

        # index to a mini-batch
        index = T.lscalar('index')
        # learning_rate = T.dscalar('learning_rate')

        # gradients1, gradients2 = T.dmatrices('gradients1', 'gradients2')   # for testing the gradients

        # gparams = [T.grad(self.classifier_cost, param)
        #            for param in self.classifier_params]

        # updates = [
        #     (param, param - learning_rate * gparam)
        #     for param, gparam in zip(self.classifier_params, gparams)
        # ]

        gparams = [T.grad(self.classifier_cost, param)
                   for param in self.params]

        if self.use_dropout:

            updates = []

            for param, gparam in zip(self.params, gparams):
                stepped_param = param - learning_rate * gparam

                if param.get_value(borrow=True).ndim == 2:
                    col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                    stepped_param = stepped_param * (desired_norms / (1e-7 + col_norms))

                updates.append((param, stepped_param))

        else:
            updates = [
                (param, param - learning_rate * gparam)
                for param, gparam in zip(self.params, gparams)
            ]

        # gradients1 = gparams[0]
        # gradients2 = gparams[2]

        # if self.classifier.name == 'svm':
        #     givens = {
        #         self.x: train_set_x[index * batch_size * 729: (index + 1) * batch_size * 729],
        #         self.y1: train_set_y[index * batch_size: (index + 1) * batch_size]
        #     }

        # else:
        #     givens = {
        #         self.x: train_set_x[index * batch_size * 729: (index + 1) * batch_size * 729],
        #         self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
        #     }

        givens = {
            self.x: train_set_x[index * batch_size * 729: (index + 1) * batch_size * 729],
            self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }

        train_fn = theano.function(
            inputs=[index, self.l2_reg, self.l1_reg],
            outputs=[self.classifier_cost, self.errors],
            updates=updates,
            givens=givens,
            name='train'
        )

        # test_score_i = theano.function(
        #     inputs=[index],
        #     outputs=self.errors,
        #     givens={
        #         self.x: test_set_x[index * batch_size * 729: (index + 1) * batch_size * 729],
        #         self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
        #     },
        #     name='test'
        # )

        # # Create a function that scans the entire test set
        # def test_score(n_batches):
        #     return [test_score_i(i) for i in xrange(n_batches)]

        return train_fn

    def build_test_function(self, dataset, batch_size):
        '''
        Build the function for the testing.
        The function 'test' that computes the error on a batch from the test set
        '''
        (test_set_x, test_set_y) = dataset

        index = T.lscalar('index')

        # if self.classifier.name == 'svm':
        #     givens = {
        #         self.x: test_set_x[index * batch_size * 729: (index + 1) * batch_size * 729],
        #         self.y1: test_set_y[index * batch_size: (index + 1) * batch_size]
        #     }

        # else:
        #     givens = {
        #         self.x: test_set_x[index * batch_size * 729: (index + 1) * batch_size * 729],
        #         self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
        #     }

        givens = {
            self.x: test_set_x[index * batch_size * 729: (index + 1) * batch_size * 729],
            self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }

        test_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens=givens,
            name='test'
        )

        # Create a function that scans the entire test set
        def test_score(n_batches):
            return [test_score_i(i) for i in xrange(n_batches)]

        return test_score


def training_function(config):
    '''
    '''
    rng = np.random.RandomState(1234)

    polarity_split = config['polarity_split']
    n_slice = config['n_slice']
    patch_width = config['patch_width']
    n_in = config['n_in']
    n_out = config['n_out']
    hidden_layer_size = config['hidden_layer_size']
    pretrain_batch_size = config['pretrain_batch_size']
    batch_size = config['finetune_batch_size']

    pretraining_epochs = config['pretraining_epochs']
    training_epochs = config['finetuning_epochs']

    NSPL = config['NSPL']
    sparsity_rate = config['sparsity_rate']
    activation = Relu if config['activation'] == 'relu' \
        or config['activation'] == 'Relu' \
        or config['activation'] == 'ReLu' \
        or config['activation'] == 'RELU' \
        else T.nnet.sigmoid

    classifier = config['classifier']
    eps = config['eps']

    pretrain_lr = config['pretrain_lr']
    finetune_lr = config['finetune_lr']
    l2_reg = config['l2_reg']
    l1_reg = config['l1_reg']

    use_dropout = config['use_dropout']
    dropout_rate = config['dropout_rate']
    max_col_norm = config['max_col_norm']

    doPretrain = config['doPretrain']
    doFinetune = config['doFinetune']

    isFirstTimePretrain = config['isFirstTimePretrain']
    isFirstTimeFinetune = config['isFirstTimeFinetune']

    parameters_dir = config['parameters_dir']

    # print the parameters to the files

    # print pretrain_patches.shape
    print '... building the model'
    with open('log', 'a+') as f:
        print >> f, '... building the model'

    # construct the network
    net = Network_epls(rng=rng,
                       n_in=n_in,
                       hidden_layer_size=hidden_layer_size,
                       n_out=n_out,
                       pretrain_mini_batch=pretrain_batch_size,
                       mini_batch=batch_size,
                       n_samples=NSPL,
                       sparsity_rate=sparsity_rate,
                       activation=activation,
                       classifier=classifier,
                       polarity_split=polarity_split,
                       use_dropout=use_dropout,
                       dropout_rate=dropout_rate,
                       max_col_norm=max_col_norm
                       )

    # save parameters of net
    # save_file = open('/mnt/UAV_Storage/richard/params.save', 'wb')
    # tmp = []
    # for param in net.params:
    #     tmp.append(param.get_value(borrow=True))
    # cPickle.dump(tmp, save_file, True)
    # save_file.close()

    print '... loading the data'
    with open('log', 'a+') as f:
        print >> f, '... loading the data'

    [(train_set_x, train_set_y), (test_set_x, test_set_y)] = load_cifar10.load_cifar_10()

    # if classifier == 'svm':
    #     train_set_y = to_categorical(train_set_y, 10)
    #     test_set_y = to_categorical(test_set_y, 10)

    temp_x = np.reshape(train_set_x, (-1, 3, 32, 32))
    temp_x = np.transpose(temp_x, (0, 3, 2, 1))

    pretrain_patches = load_cifar10.extract_patches_for_pretrain(dataset=temp_x,
                                                                 NSPL=NSPL,
                                                                 patch_width=patch_width
                                                                 )

    pretrain_patches = load_cifar10.global_contrast_normalize(pretrain_patches)
    # local contrast normalize
    data_mean = pretrain_patches.mean(axis=0)
    # pretrain_patches -= data_mean

    normalizers = np.sqrt(0.01 + pretrain_patches.var(axis=0, ddof=1))
    normalizers[normalizers < 1e-8] = 1.
    # pretrain_patches /= normalizers

    # pretrain_patches = load_cifar10.global_contrast_normalize(pretrain_patches)
    # pretrain_patches = load_cifar10.zca(pretrain_patches)
    if doPretrain:

        pretrain_patches -= data_mean
        pretrain_patches /= normalizers
        pretrain_patches = load_cifar10.zca(pretrain_patches)

        pretrain_patches = np.asarray(pretrain_patches, dtype=np.float32)

        n_samples = pretrain_patches.shape[0]

        shared_pretrain_patches = load_cifar10.shared_dataset_x(pretrain_patches)

        n_train_batches = int(NSPL / pretrain_batch_size)

        if isFirstTimePretrain:

            with open('pretrain_data.csv', 'a+') as f:
                mywrite = csv.writer(f)
                mywrite.writerow(['pretrain_loss',
                                  'NSPL_{}'.format(NSPL),
                                  'sparsity_{}'.format(sparsity_rate),
                                  'hidden_{}'.format(hidden_layer_size),
                                  'batchsize_{}'.format(pretrain_batch_size),
                                  'lr_{}'.format(pretrain_lr)
                                  ]
                                 )

            print 'The first time to pretrain!'
            with open('log', 'a+') as f:
                print >> f, 'The first time to pretrain!'

            #########################
            # PRETRAINING THE MODEL #
            #########################
            print '... getting the pretraining functions'
            with open('log', 'a+') as f:
                print >> f, '... getting the pretraining functions'

            pretraining_fn = net.pretraining_function(
                train_set_x=shared_pretrain_patches,
                pretrain_batch_size=pretrain_batch_size,
                learning_rate=pretrain_lr
            )

            print '... pre-training the model'
            with open('log', 'a+') as f:
                print >> f, '... pre-training the model'

            start_time = time.clock()

            epoch = 0
            done_looping = False
            epoch_error = []

            while (epoch < pretraining_epochs) and (not done_looping):

                # save the epoch
                with open(parameters_dir + 'current_pretrain_epoch.save', 'wb') as f:
                    cPickle.dump(epoch, f, True)

                # save_file = open('/mnt/UAV_Storage/richard/pretrain_epoch.save', 'wb')
                # cPickle.dump(epoch, save_file, True)
                # save_file.close()

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

                with open('pretrain_data.csv', 'a+') as f:
                    mywrite = csv.writer(f)
                    mywrite.writerow([loss])

                print 'The loss of epoch {0} is {1}'.format(epoch, loss)
                with open('log', 'a+') as f:
                    print >> f, 'The loss of epoch {0} is {1}'.format(epoch, loss)

                epoch_error.append(loss)

                np.savetxt(parameters_dir + 'hiddenLayer_output.txt', minibatch_output, fmt='%f', delimiter=',')

                # save the params of pretraining
                with open(parameters_dir + 'current_pretrain_params.save', 'wb') as f:
                    temp = []
                    for param in net.pretrain_params:
                        temp.append(param.get_value(borrow=True))
                    cPickle.dump(temp, f, True)

                # save_file = open('/mnt/UAV_Storage/richard/pretrain_params.save', 'wb')
                # temp = []
                # for param in net.pretrain_params:
                #     temp.append(param.get_value(borrow=True))
                # cPickle.dump(temp, save_file, True)
                # save_file.close()

                # save the params of network
                with open(parameters_dir + 'current_params.save', 'wb') as f:
                    temp = []
                    for param in net.params:
                        temp.append(param.get_value(borrow=True))
                    cPickle.dump(temp, f, True)

                # save_file = open('/mnt/UAV_Storage/richard/params.save', 'wb')
                # temp = []
                # for param in net.params:
                #     temp.append(param.get_value(borrow=True))
                # cPickle.dump(temp, save_file, True)
                # save_file.close()

                # stop condition
                # The relative decrement error between epochs is smaller than eps
                if len(epoch_error) > 1:
                    err = (epoch_error[-2] - epoch_error[-1]) / epoch_error[-2]
                    if err < eps:
                        done_looping = True

                epoch = epoch + 1

            end_time = time.clock()

            print >> sys.stderr, ('The pretraining code for file ' +
                                  os.path.split(__file__)[1] +
                                  ' ran for %.2fm' % ((end_time - start_time) / 60.))

        else:
            print 'Continue to pretrain!'
            with open('log', 'a+') as f:
                print >> f, 'Continue to pretrain!'

            # load the pretrain params of network
            with open(parameters_dir + 'current_pretrain_params.save') as f:
                pas = cPickle.load(f)
                params = [np.asarray(pa, dtype=theano.config.floatX) for pa in pas]
                for pa, param in zip(params, net.pretrain_params):
                    param.set_value(pa)

            # save_file = open('/mnt/UAV_Storage/richard/params.save')
            # pas = cPickle.load(save_file)
            # params = [np.asarray(pa, dtype=theano.config.floatX) for pa in pas]

            # for pa, param in zip(params, net.params):
            #     param.set_value(pa)

            # load the epoch
            with open(parameters_dir + 'current_pretrain_epoch.save') as f:
                epoch = cPickle.load(f)

            #########################
            # PRETRAINING THE MODEL #
            #########################
            print '... getting the pretraining functions'
            with open('log', 'a+') as f:
                print >> f, '... getting the pretraining functions'

            pretraining_fn = net.pretraining_function(
                train_set_x=shared_pretrain_patches,
                pretrain_batch_size=pretrain_batch_size,
                learning_rate=pretrain_lr
            )

            print '... pre-training the model'
            with open('log', 'a+') as f:
                print >> f, '... pre-training the model'

            start_time = time.clock()

            done_looping = False
            epoch_error = []

            while (epoch < pretraining_epochs) and (not done_looping):

                # save the epoch
                with open(parameters_dir + 'current_pretrain_epoch.save', 'wb') as f:
                    cPickle.dump(epoch, f, True)

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

                with open('pretrain_data.csv', 'a+') as f:
                    mywrite = csv.writer(f)
                    mywrite.writerow([loss])

                print 'The loss of epoch {0} is {1}'.format(epoch, loss)
                with open('log', 'a+') as f:
                    print >> f, 'The loss of epoch {0} is {1}'.format(epoch, loss)

                epoch_error.append(loss)

                np.savetxt(parameters_dir + 'hiddenLayer_output.txt', minibatch_output, fmt='%f', delimiter=',')

                # save the params of pretraining
                with open(parameters_dir + 'current_pretrain_params.save', 'wb') as f:
                    temp = []
                    for param in net.pretrain_params:
                        temp.append(param.get_value(borrow=True))
                    cPickle.dump(temp, f, True)

                # save_file = open('/mnt/UAV_Storage/richard/pretrain_params.save', 'wb')
                # temp = []
                # for param in net.pretrain_params:
                #     temp.append(param.get_value(borrow=True))
                # cPickle.dump(temp, save_file, True)
                # save_file.close()

                # save the params of network
                with open(parameters_dir + 'current_params.save', 'wb') as f:
                    temp = []
                    for param in net.params:
                        temp.append(param.get_value(borrow=True))
                    cPickle.dump(temp, f, True)

                # save_file = open('/mnt/UAV_Storage/richard/params.save', 'wb')
                # temp = []
                # for param in net.params:
                #     temp.append(param.get_value(borrow=True))
                # cPickle.dump(temp, save_file, True)
                # save_file.close()

                # stop condition
                # The relative decrement error between epochs is smaller than eps
                if len(epoch_error) > 1:
                    err = (epoch_error[-2] - epoch_error[-1]) / epoch_error[-2]
                    if err < eps:
                        done_looping = True

                epoch = epoch + 1

            end_time = time.clock()

            print >> sys.stderr, ('The pretraining code for file ' +
                                  os.path.split(__file__)[1] +
                                  ' ran for %.2fm' % ((end_time - start_time) / 60.))

    if doFinetune:

        ###########################
        # TRAINING THE CLASSIFIER #
        ###########################
        print '... training the classifier'
        with open('log', 'a+') as f:
            print >> f, '... training the classifier'

        ntrain = train_set_y.shape[0]
        ntest = test_set_y.shape[0]

        # shared_train_x = theano.shared(np.ones((14 * 14 * ntrain, n_in), dtype=theano.config.floatX), borrow=True)
        # shared_test_x = theano.shared(np.ones((14 * 14 * ntest, n_in), dtype=theano.config.floatX), borrow=True)

        print '... Preparing the data'
        with open('log', 'a+') as f:
            print >> f, '... Preparing the data'

        temp_x = np.reshape(train_set_x, (-1, 3, 32, 32))
        temp_x = np.transpose(temp_x, (0, 3, 2, 1))

        # train_patches = load_cifar10.extract_patches_for_classification(dataset=temp_x,
        #                                                                 n_patches_per_image=14 * 14,
        #                                                                 patch_width=patch_width
        #                                                                 )
        train_patches = load_cifar10.extract_patches_for_classification(dataset=temp_x,
                                                                        patch_width=patch_width
                                                                        )

        train_patches = load_cifar10.global_contrast_normalize(train_patches)

        # local contrast normalize
        # train_data_mean = train_patches.mean(axis=0)
        train_patches -= data_mean
        # normalizers = np.sqrt(10 + train_patches.var(axis=0, ddof=1))
        # normalizers[normalizers < 1e-8] = 1.
        train_patches /= normalizers
        train_patches = load_cifar10.zca(train_patches)

        # shared_train_x = load_cifar10.shared_dataset_x(train_patches)
        # shared_train_y = load_cifar10.shared_dataset_y(train_set_y)

        temp_x = np.reshape(test_set_x, (-1, 3, 32, 32))
        temp_x = np.transpose(temp_x, (0, 3, 2, 1))

        # test_patches = load_cifar10.extract_patches_for_classification(dataset=temp_x,
        #                                                                n_patches_per_image=14 * 14,
        #                                                                patch_width=patch_width
        #                                                                )
        test_patches = load_cifar10.extract_patches_for_classification(dataset=temp_x,
                                                                       patch_width=patch_width
                                                                       )

        test_patches = load_cifar10.global_contrast_normalize(test_patches)

        test_patches -= data_mean
        test_patches /= normalizers
        test_patches = load_cifar10.zca(test_patches)

        # shared_test_x = load_cifar10.shared_dataset_x(test_patches)
        # shared_test_y = load_cifar10.shared_dataset_y(test_set_y)

        # train_set = (shared_train_x, shared_train_y)
        # test_set = (shared_test_x, shared_test_y)
        # datasets = (train_set, test_set)

        n_train_batches = int(ntrain / batch_size)
        # n_test_batches = int(ntest / batch_size)

        start_time = timeit.default_timer()

        # Split the dataset
        # The origin dataset is too big to the GPU
        # n_slice = 10  # split the dataset to n_slice parts
        n_samples_of_every_slice = int(ntrain / n_slice)
        n_batches_of_slice = int(n_samples_of_every_slice / batch_size)
        n_test_batches = int(ntest / batch_size)

        shared_train_x = theano.shared(np.ones((n_samples_of_every_slice * 27 * 27, n_in), dtype=theano.config.floatX), borrow=True)

        # shared_test_x = theano.shared(np.ones((ntest * 27 * 27, n_in), dtype=theano.config.floatX), borrow=True)

        best_test_error = np.inf

        if isFirstTimeFinetune:

            print 'The first time to finetune!'
            with open('log', 'a+') as f:
                print >> f, 'The first time to finetune!'

            # load the pretrain parameters
            if os.path.exists(parameters_dir + 'current_pretrain_params.save'):

                with open(parameters_dir + 'current_pretrain_params.save') as f:
                    pas = cPickle.load(f)

                    if use_dropout:
                        params = [np.asarray(pa, dtype=theano.config.floatX) / (1 - dropout_rate) for pa in pas]
                    else:
                        params = [np.asarray(pa, dtype=theano.config.floatX) for pa in pas]

                    for pa, param in zip(params, net.pretrain_params):
                        param.set_value(pa)

                    # for ind in range(len(pas)):
                    #     pa = pas[ind]
                    #     net.pretrain_params[ind].set_value(np.asarray(pa, dtype=theano.config.floatX))

            else:
                print 'No pretrain before finetune!'
                with open('log', 'a+') as f:
                    print >> f, 'No pretrain before finetune!'

                # save_file = open('/mnt/UAV_Storage/richard/pretrain_params.save')
                # pas = cPickle.load(save_file)

                # for ind in range(len(pas)):
                #     pa = pas[ind]
                #     net.params[ind].set_value(np.asarray(pa, dtype=theano.config.floatX))

                # params = [np.asarray(pa, dtype=theano.config.floatX) for pa in pas]

                # for pa, param in zip(params, net.params):
                #     param.set_value(pa)

            for i in range(len(finetune_lr)):

                with open('finetune_data.csv', 'a+') as f:
                    mywrite = csv.writer(f)
                    mywrite.writerow(['test_error',
                                      'train_error',
                                      'train_loss',
                                      'lr_{}'.format(finetune_lr[i]),
                                      'polaritySplit_{}'.format(polarity_split),
                                      'hidden_{}'.format(hidden_layer_size),
                                      'batchsize_{}'.format(batch_size),
                                      'weightDecay_{}'.format(l2_reg),
                                      'useDropout_{}'.format(use_dropout),
                                      'dropout_{}'.format(dropout_rate),
                                      'maxColNorm_{}'.format(max_col_norm)
                                      ]
                                     )

                print '...... for finetune_learning_rate_{0}: {1}'.format(i, finetune_lr[i])
                with open('log', 'a+') as f:
                    print >> f, '...... for finetune_learning_rate_{0}: {1}'.format(i, finetune_lr[i])

                print '... getting the finetuning functions'
                with open('log', 'a+') as f:
                    print >> f, '... getting the finetuning functions'

                # save the learning rate
                with open(parameters_dir + 'current_finetune_lr.save', 'wb') as f:
                    cPickle.dump(finetune_lr[i], f, True)

                # save_file = open('/mnt/UAV_Storage/richard/current_finetune_lr.save', 'wb')
                # cPickle.dump(finetune_lr[i], save_file, True)
                # save_file.close()

                epoch = 0
                while (epoch < training_epochs):
                    print 'Training epoch {0}'.format(epoch)
                    with open('log', 'a+') as f:
                        print >> f, 'Training epoch {0}'.format(epoch)

                    # save the current finetune epoch
                    with open(parameters_dir + 'current_finetune_epoch.save', 'wb') as f:
                        cPickle.dump(epoch, f, True)

                    # epoch += 1
                    if use_dropout:
                        DropoutLayer.SetDropoutOn()
                    # print 'dropout flag on: {}'.format(net.dropout.flag_on.get_value())

                    train_errors = []
                    loss = []

                    for j in range(n_slice):
                        shared_train_x.set_value(train_patches[j * n_samples_of_every_slice * 27 * 27:(j + 1) * n_samples_of_every_slice * 27 * 27], borrow=True)
                        shared_train_y = load_cifar10.shared_dataset_y(train_set_y[j * n_samples_of_every_slice:(j + 1) * n_samples_of_every_slice])

                        # shared_test_x.set_value(test_patches, borrow=True)
                        # shared_test_y = load_cifar10.shared_dataset_y(test_set_y)

                        # datasets = ((shared_train_x, shared_train_y), (shared_test_x, shared_test_y))

                        train_fn = net.build_finetune_function(datasets=(shared_train_x, shared_train_y),
                                                               batch_size=batch_size,
                                                               learning_rate=finetune_lr[i]
                                                               )

                        for minibatch_index in xrange(n_batches_of_slice):
                            minibatch_avg_cost, minibatch_error = train_fn(minibatch_index, l2_reg, l1_reg)
                            train_errors.append(minibatch_error)
                            loss.append(minibatch_avg_cost)

                    print 'Epoch {0}, Mean loss: {1}'.format(epoch, np.mean(loss))
                    print 'Epoch {0}, Minimum loss: {1}'.format(epoch, np.min(loss))
                    with open('log', 'a+') as f:
                        print >> f, 'Epoch {0}, Mean loss: {1}'.format(epoch, np.mean(loss))
                        print >> f, 'Epoch {0}, Minimum loss: {1}'.format(epoch, np.min(loss))

                    print 'Epoch {0}, Mean train error {1}%'.format(epoch, np.mean(train_errors) * 100)
                    print 'Epoch {0}, Minimum train error {1}%'.format(epoch, np.min(train_errors) * 100)
                    with open('log', 'a+') as f:
                        print >> f, 'Epoch {0}, Mean train error {1}%, {2}'.format(epoch, np.mean(train_errors) * 100, np.mean(train_errors))
                        print >> f, 'Epoch {0}, Minimum train error {1}%'.format(epoch, np.min(train_errors) * 100)

                    print '... Testing...'
                    with open('log', 'a+') as f:
                        print >> f, '... Testing...'

                    shared_test_x = load_cifar10.shared_dataset_x(test_patches)
                    shared_test_y = load_cifar10.shared_dataset_y(test_set_y)

                    if use_dropout:
                        DropoutLayer.SetDropoutOff()
                    # print 'dropout flag on: {}'.format(net.dropout.flag_on.get_value())

                    test_model = net.build_test_function(dataset=(shared_test_x, shared_test_y),
                                                         batch_size=batch_size
                                                         )

                    test_error = test_model(n_test_batches)
                    this_test_error = np.mean(test_error)
                    this_min_error = np.min(test_error)

                    print '... Epoch {0}, Mean test error {1}%'.format(epoch, this_test_error * 100)
                    print '... Epoch {0}, Minimum test error {1}%'.format(epoch, this_min_error * 100)
                    with open('log', 'a+') as f:
                        print >> f, '... Epoch {0}, Mean test error {1}%, {2}'.format(epoch, this_test_error * 100, this_test_error)
                        print >> f, '... Epoch {0}, Minimum test error {1}%'.format(epoch, this_min_error * 100)

                    if this_test_error < best_test_error:
                        best_test_error = this_test_error

                    with open('finetune_data.csv', 'a+') as f:
                        mywrite = csv.writer(f)
                        mywrite.writerow([this_test_error, np.mean(train_errors), np.mean(loss)])

                    with open(parameters_dir + 'current_params.save', 'wb') as f:
                        temp = []
                        for param in net.params:
                            temp.append(param.get_value(borrow=True))
                        cPickle.dump(temp, f, True)

                    # save_file = open('/mnt/UAV_Storage/richard/params.save', 'wb')
                    # tmp = []
                    # for param in net.params:
                    #     tmp.append(param.get_value(borrow=True))
                    # cPickle.dump(tmp, save_file, True)
                    # save_file.close()

                    epoch += 1

            end_time = timeit.default_timer()

            print >> sys.stderr, ('The training code for file ' +
                                  os.path.split(__file__)[1] +
                                  ' ran for %.2fm' % ((end_time - start_time) / 60.))
        else:

            print 'Continue to finetune!'
            with open('log', 'a+') as f:
                print >> f, 'Continue to finetune!'

            # load the parameters
            with open(parameters_dir + 'current_params.save') as f:
                pas = cPickle.load(f)
                params = [np.asarray(pa, dtype=theano.config.floatX) for pa in pas]

                for pa, param in zip(params, net.params):
                    param.set_value(pa)

            # save_file = open('/mnt/UAV_Storage/richard/current_params.save')
            # pas = cPickle.load(save_file)
            # params = [np.asarray(pa, dtype=theano.config.floatX) for pa in pas]

            # for pa, param in zip(params, net.params):
            #     param.set_value(pa)

            # load current epoch
            with open(parameters_dir + 'current_finetune_epoch.save') as f:
                epoch = cPickle.load(f)

            # save_file = open('/mnt/UAV_Storage/richard/current_finetune_epoch.save')
            # epoch = cPickle.load(save_file)
            # save_file.close()

            # load current learning rate
            with open(parameters_dir + 'current_finetune_lr.save') as f:
                current_finetune_lr = cPickle.load(f)

            # save_file = open('/mnt/UAV_Storage/richard/current_finetune_lr.save')
            # current_finetune_lr = cPickle.load(save_file)
            # save_file.close()

            if current_finetune_lr in finetune_lr:
                ind = finetune_lr.index(current_finetune_lr)
                finetune_lr = finetune_lr[ind:]
            else:
                print 'The current fine tune learn rate is not in finetune_lr!'

            for i in range(len(finetune_lr)):

                if epoch == 0:
                    with open('finetune_data.csv', 'a+') as f:
                        mywrite = csv.writer(f)
                        mywrite.writerow(['test_error',
                                          'train_error',
                                          'train_loss',
                                          'lr_{}'.format(finetune_lr[i]),
                                          'polaritySplit_{}'.format(polarity_split),
                                          'hidden_{}'.format(hidden_layer_size),
                                          'batchsize_{}'.format(batch_size),
                                          'weightDecay_{}'.format(l2_reg),
                                          'useDropout_{}'.format(use_dropout),
                                          'dropout_{}'.format(dropout_rate),
                                          'maxColNorm_{}'.format(max_col_norm)
                                          ]
                                         )

                print '...... for finetune_learning_rate_{0}: {1}'.format(i, finetune_lr[i])
                print '... getting the finetuning functions'

                with open('log', 'a+') as f:
                    print >> f, '...... for finetune_learning_rate_{0}: {1}'.format(i, finetune_lr[i])
                    print >> f, '... getting the finetuning functions'

                # save the learning rate
                with open(parameters_dir + 'current_finetune_lr.save', 'wb') as f:
                    cPickle.dump(finetune_lr[i], f, True)

                # save_file = open('/mnt/UAV_Storage/richard/current_finetune_lr.save', 'wb')
                # cPickle.dump(finetune_lr[i], save_file, True)
                # save_file.close()

                # epoch = 0

                while (epoch < training_epochs):
                    print 'Training epoch {0}'.format(epoch)
                    with open('log', 'a+') as f:
                        print >> f, 'Training epoch {0}'.format(epoch)

                    # save the current finetune epoch
                    with open(parameters_dir + 'current_finetune_epoch.save', 'wb') as f:
                        cPickle.dump(epoch, f, True)

                    # save_file = open('/mnt/UAV_Storage/richard/current_finetune_epoch.save', 'wb')
                    # cPickle.dump(epoch, save_file, True)
                    # save_file.close()

                    # epoch += 1
                    if use_dropout:
                        DropoutLayer.SetDropoutOn()
                    # print 'dropout flag on: {}'.format(net.dropout.flag_on.get_value())

                    train_errors = []
                    loss = []

                    for j in range(n_slice):
                        shared_train_x.set_value(train_patches[j * n_samples_of_every_slice * 27 * 27:(j + 1) * n_samples_of_every_slice * 27 * 27], borrow=True)
                        shared_train_y = load_cifar10.shared_dataset_y(train_set_y[j * n_samples_of_every_slice:(j + 1) * n_samples_of_every_slice])

                        # shared_test_x.set_value(test_patches, borrow=True)
                        # shared_test_y = load_cifar10.shared_dataset_y(test_set_y)

                        # datasets = ((shared_train_x, shared_train_y), (shared_test_x, shared_test_y))

                        train_fn = net.build_finetune_function(datasets=(shared_train_x, shared_train_y),
                                                               batch_size=batch_size,
                                                               learning_rate=finetune_lr[i]
                                                               )

                        for minibatch_index in xrange(n_batches_of_slice):
                            minibatch_avg_cost, minibatch_error = train_fn(minibatch_index, l2_reg, l1_reg)
                            train_errors.append(minibatch_error)
                            loss.append(minibatch_avg_cost)

                    print 'Epoch {0}, Mean loss: {1}'.format(epoch, np.mean(loss))
                    print 'Epoch {0}, Minimum loss: {1}'.format(epoch, np.min(loss))
                    with open('log', 'a+') as f:
                        print >> f, 'Epoch {0}, Mean loss: {1}'.format(epoch, np.mean(loss))
                        print >> f, 'Epoch {0}, Minimum loss: {1}'.format(epoch, np.min(loss))

                    print 'Epoch {0}, Mean train error {1}%'.format(epoch, np.mean(train_errors) * 100)
                    print 'Epoch {0}, Minimum train error {1}%'.format(epoch, np.min(train_errors) * 100)
                    with open('log', 'a+') as f:
                        print >> f, 'Epoch {0}, Mean train error {1}%, {2}'.format(epoch, np.mean(train_errors) * 100, np.mean(train_errors))
                        print >> f, 'Epoch {0}, Minimum train error {1}%'.format(epoch, np.min(train_errors) * 100)

                    print '... Testing...'
                    with open('log', 'a+') as f:
                        print >> f, '... Testing...'

                    shared_test_x = load_cifar10.shared_dataset_x(test_patches)
                    shared_test_y = load_cifar10.shared_dataset_y(test_set_y)

                    if use_dropout:
                        DropoutLayer.SetDropoutOff()
                    # print 'dropout flag on: {}'.format(net.dropout.flag_on.get_value())

                    test_model = net.build_test_function(dataset=(shared_test_x, shared_test_y),
                                                         batch_size=batch_size
                                                         )

                    test_error = test_model(n_test_batches)
                    this_test_error = np.mean(test_error)
                    this_min_error = np.min(test_error)
                    print '... Epoch {0}, Mean test error {1}%'.format(epoch, this_test_error * 100)
                    print '... Epoch {0}, Minimum test error {1}%'.format(epoch, this_min_error * 100)
                    with open('log', 'a+') as f:
                        print >> f, '... Epoch {0}, Mean test error {1}%, {2}'.format(epoch, this_test_error * 100, this_test_error)
                        print >> f, '... Epoch {0}, Minimum test error {1}%'.format(epoch, this_min_error * 100)

                    if this_test_error < best_test_error:
                        best_test_error = this_test_error

                    with open('finetune_data.csv', 'a+') as f:
                        mywrite = csv.writer(f)
                        mywrite.writerow([this_test_error, np.mean(train_errors), np.mean(loss)])

                    with open(parameters_dir + 'current_params.save', 'wb') as f:
                        tmp = []
                        for param in net.params:
                            tmp.append(param.get_value(borrow=True))
                        cPickle.dump(tmp, f, True)

                    # save_file = open('/mnt/UAV_Storage/richard/params.save', 'wb')
                    # tmp = []
                    # for param in net.params:
                    #     tmp.append(param.get_value(borrow=True))
                    # cPickle.dump(tmp, save_file, True)
                    # save_file.close()

                    epoch += 1

                epoch = 0

            end_time = timeit.default_timer()

            print >> sys.stderr, ('The training code for file ' +
                                  os.path.split(__file__)[1] +
                                  ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    training_function(config)

    print 'DONE!'
