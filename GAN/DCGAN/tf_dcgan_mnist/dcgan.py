from __future__ import division

import numpy as np
import tensorflow as tf
from utils import Conv2dLayer
from utils import DeconvLayer
from utils import BatchNormLayer


def generator(z, is_training=True):
    batch_size = z.get_shape().as_list()[0]
    with tf.variable_scope('generator'):
        z = tf.reshape(z, [batch_size, 1, 1, -1])
        # conv1 = DeconvLayer(z,)


def discriminator(inputs, is_training, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        pass
