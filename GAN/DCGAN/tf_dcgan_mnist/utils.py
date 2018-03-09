from __future__ import division

import numpy as np
import tensorflow as tf


class Conv2dLayer(object):
    """Construct conv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 activation=tf.identity,
                 padding='SAME',
                 stride=1,
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = tf.get_variable(name='w',
                                     shape=filter_shape,
                                     initializer=tf.glorot_normal_initializer())

            self.b = tf.get_variable(name='b',
                                     shape=filter_shape[-1],
                                     initializer=tf.constant_initializer(0.))

            linear_output = tf.nn.conv2d(self.inputs,
                                         self.w,
                                         [1, stride, stride, 1],
                                         padding=padding)
            self.output = activation(tf.nn.bias_add(linear_output, self.b))
            self.output_shape = self.output.get_shape().as_list()


class DeconvLayer(object):
    """Construct deconv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 output_shape,
                 activation=tf.identity,
                 padding='SAME',
                 stride=1,
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = tf.get_variable(name='w',
                                     shape=filter_shape,
                                     initializer=tf.glorot_normal_initializer())

            self.b = tf.get_variable(name='b',
                                     shape=filter_shape[-1],
                                     initializer=tf.constant_initializer(0.))

            deconv = tf.nn.conv2d_transpose(value=self.inputs,
                                            filter=self.w,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding=padding)

            self.output = activation(tf.nn.bias_add(deconv, self.b))
            self.output_shape = self.output.get_shape().as_list()


class DilatedConv2dLayer(object):
    """Construct dilated conv2d layer."""

    def __init__(self,
                 inputs,
                 filter_shape,
                 rate,
                 activation=tf.identity,
                 padding='SAME',
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.w = tf.get_variable(name='w',
                                     shape=filter_shape,
                                     initializer=tf.glorot_normal_initializer())

            self.b = tf.get_variable(name='b',
                                     shape=filter_shape[-1],
                                     initializer=tf.constant_initializer(0.))

            linear_output = tf.nn.atrous_conv2d(value=self.inputs,
                                                filters=self.w,
                                                rate=rate,
                                                padding=padding)
            self.output = activation(tf.nn.bias_add(linear_output, self.b))
            self.output_shape = self.output.get_shape().as_list()


class FCLayer(object):
    """Construct FC layer."""

    def __init__(self,
                 inputs,
                 output_size,
                 activation=tf.identity,
                 name=None):
        self.inputs = inputs
        shape = inputs.get_shape().as_list()
        input_size = np.prod(shape[1:])
        x = tf.reshape(self.inputs, [-1, input_size])

        with tf.variable_scope(name):
            self.w = tf.get_variable(name='w',
                                     shape=[input_size, output_size],
                                     initializer=tf.glorot_normal_initializer())
            self.b = tf.get_variable(name='b',
                                     shape=[output_size],
                                     initializer=tf.constant_initializer(0.))

            self.output = activation(tf.nn.bias_add(tf.matmul(x, self.w), self.b))
            self.output_shape = self.output.get_shape().as_list()


class BatchNormLayer(object):
    """Construct batch norm layer."""

    def __init__(self,
                 inputs,
                 is_training,
                 decay=0.999,
                 epsilon=1e-5,
                 name=None):
        self.inputs = inputs
        with tf.variable_scope(name):
            self.scale = tf.get_variable(name='scale',
                                         shape=[inputs.get_shape()[-1]],
                                         initializer=tf.constant_initializer(1.))
            self.beta = tf.get_variable(name='beta',
                                        shape=[inputs.get_shape()[-1]],
                                        initializer=tf.constant_initializer(0.))
            self.pop_mean = tf.get_variable(name='pop_mean',
                                            shape=[inputs.get_shape()[-1]],
                                            initializer=tf.constant_initializer(0.),
                                            trainable=False)
            self.pop_var = tf.get_variable(name='pop_var',
                                           shape=[inputs.get_shape()[-1]],
                                           initializer=tf.constant_initializer(1.),
                                           trainable=False)

            def mean_var_update():
                axes = list(range(len(inputs.get_shape()) - 1))
                batch_mean, batch_var = tf.nn.moments(inputs, axes)
                train_mean = tf.assign(self.pop_mean, self.pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(self.pop_var, self.pop_var * decay + batch_var * (1 - decay))

                with tf.control_dependencies([train_mean, train_var]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, variance = tf.cond(is_training, mean_var_update, lambda: (self.pop_mean, self.pop_var))
            self.output = tf.nn.batch_normalization(x=inputs,
                                                    mean=mean,
                                                    variance=variance,
                                                    offset=self.beta,
                                                    scale=self.scale,
                                                    variance_epsilon=epsilon)
