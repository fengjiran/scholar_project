from __future__ import division

import numpy as np
import tensorflow as tf
from utils import Conv2dLayer
from utils import DeconvLayer
from utils import BatchNormLayer
from utils import FCLayer


def generator(z, is_training):
    batch_size = z.get_shape().as_list()[0]
    with tf.variable_scope('generator'):
        fc1 = FCLayer(inputs=z, output_size=256 * 7 * 7, name='fc1')
        bn1 = BatchNormLayer(fc1.output, is_training, name='bn1')
        bn1 = tf.nn.relu(bn1.output)
        bn1 = tf.reshape(bn1, [batch_size, 7, 7, 256])

        deconv2 = DeconvLayer(inputs=bn1,
                              filter_shape=[5, 5, 128, 256],
                              output_shape=[batch_size, 14, 14, 128],
                              stride=2,
                              name='deconv2')
        bn2 = BatchNormLayer(deconv2.output, is_training, name='bn2')
        bn2 = tf.nn.relu(bn2.output)

        deconv3 = DeconvLayer(inputs=bn2,
                              filter_shape=[5, 5, 64, 128],
                              output_shape=[batch_size, 28, 28, 64],
                              stride=2,
                              name='deconv3')
        bn3 = BatchNormLayer(deconv3.output, is_training, name='bn3')
        bn3 = tf.nn.relu(bn3.output)

        deconv4 = DeconvLayer(inputs=bn3,
                              filter_shape=[5, 5, 32, 64],
                              output_shape=[batch_size, 28, 28, 32],
                              name='deconv4')
        bn4 = BatchNormLayer(deconv4.output, is_training, name='bn4')
        bn4 = tf.nn.relu(bn4.output)

        deconv5 = DeconvLayer(inputs=bn4,
                              filter_shape=[5, 5, 1, 32],
                              output_shape=[batch_size, 28, 28, 1],
                              name='deconv5')
        output = tf.nn.tanh(deconv5.output)
        print(output.get_shape())
        return output

        # print(deconv2.output_shape)
        # z = tf.reshape(z, [batch_size, 1, 1, -1])
        # conv1 = DeconvLayer(z,)


def discriminator(inputs, is_training, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        pass


if __name__ == '__main__':
    train_flag = tf.placeholder(tf.bool)
    z = tf.random_uniform([100, 100], minval=-1, maxval=1, dtype=tf.float32)
    # with tf.Session() as sess:
    #     sess.run(generator(z))
    generator(z, train_flag)
