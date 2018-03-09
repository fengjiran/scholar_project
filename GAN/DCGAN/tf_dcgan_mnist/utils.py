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
