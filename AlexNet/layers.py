from __future__ import division

import os
import sys

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d


def Relu(x):
    return T.maximum(0, x)

rng = np.random.RandomState(23455)


class Weight(object):

    def __init__(self, w_shape, mean=0, std=0.01):
        super(Weight, self).__init__()
        if std != 0:
            self.np_values = np.asarray(
                rng.normal(mean, std, w_shape),
                dtype=theano.config.floatX
            )
        else:
            self.np_values = np.cast[theano.config.floatX](
                mean * np.ones(w_shape, dtype=theano.config.floatX)
            )

        self.val = theano.shared(value=self.np_values, borrow=True)

    def save_weight(self, dir, name):
        print 'weight saved: ' + name
        np.save(dir + name + '.npy', self.val.get_value())

    def load_weight(self, dir, name):
        print 'weight loaded: ' + name
        self.np_values = np.load(dir + name + '.npy')
        self.val.set_value(self.np_values)


class DataLayer(object):

    def __init__(self, input, image_shape, cropsize, rand, mirror, flag_rand):
        '''
        The random mirroring and cropping in this function is done for the whole batch.
        '''

        # trick for random mirroring
        mirror_ = input[:, :, :, ::-1]
        input = T.concatenate([input, mirror_], axis=1)

        # crop images
        center_margin = (image_shape[3] - cropsize) / 2

        if flag_rand:
            mirror_rand = T.cast(rand[2], 'int32')
            crop_xs = T.cast(rand[0] * center_margin * 2, 'int32')
            crop_ys = T.cast(rand[1] * center_margin * 2, 'int32')

        else:
            mirror_rand = 0
            crop_xs = center_margin
            crop_ys = center_margin

        self.output = input[:, mirror_rand * 3:(mirror_rand + 1) * 3, :, :]
        self.output = self.output[:, :, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize]

        print 'Data layer with shape_in: ' + str(image_shape)


class ConvPoolLayer(object):

    def __init__(self, input, input_shape, filter_shape, conv_stride, padding,
                 group, pool_size, pool_stride, bias_init, lrn=False
                 ):
        '''
        lib_conv can be cudnn (recommended) or cudaconvnet
        '''
        self.filter_size = filter_shape
        self.conv_stride = conv_stride
        self.padding = padding
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.channel = input_shape[1]
        self.lrn = lrn
        # self.lib_conv = lib_conv

        assert group in [1, 2]

        self.filter_shape = np.asarray(filter_shape)
        self.input_shape = np.asarray(input_shape)

        self.W = Weight(self.filter_shape)
        self.b = Weight(self.filter_shape[0], bias_init, std=0)

        # Convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W.val,
            input_shape=self.input_shape,
            filter_shape=self.filter_shape,
            border_mode=padding,
            subsample=(conv_stride, conv_stride)
        )

        conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')

        # Relu
        self.output = Relu(conv_out)

        # Pooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=pool_size,
            ignore_border=True,
            st=(pool_stride, pool_stride) if pool_stride else None
        )

        self.output = pooled_out

        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']

        print 'Conv layer with shape_in: {0}'.format(str(input_shape))


class FCLayer(object):

    def __init__(self, input, n_in, n_out):
        '''
        '''
        w_shape = (n_in, n_out)
        self.W = Weight(w_shape, std=0.005)
        self.b = Weight((n_out,), mean=0.1, std=0)

        self.input = input

        lin_output = T.dot(self.input, self.W.val) + self.b.val

        # Relu
        self.output = Relu(lin_output)

        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']

        print 'FC layer with num_in: ' + str(n_in) + 'num_out: ' + str(n_out)


class DropoutLayer(object):

    seed_common = np.random.RandomState(0)
    layers = []

    def __init__(self, input, n_in, n_out, prob_drop=0.5):

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


class SoftmaxLayer(object):

    def __init__(self, input, n_in, n_out):
        '''
        '''
        w_shape = (n_in, n_out)
        self.W = Weight(w_shape)
        self.b = Weight((n_out,), std=0)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W.val) + self.b.val)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']

        print 'Softmax layer with num_in: ' + str(n_in) + 'num_out: ' + str(n_out)

    def negetive_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
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

    def errors_top_x(self, y, num_top=5):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            y_pred_top_x = T.argsort(self.p_y_given_x, axis=1)[:, -num_top:]
            y_top_x = y.reshape((y.shape[0], 1)).repeat(num_top, axis=1)
            return T.mean(T.min(T.neq(y_pred_top_x, y_top_x), axis=1))
        else:
            raise NotImplementedError()
