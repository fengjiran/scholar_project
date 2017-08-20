from __future__ import division

import sys

import theano
import theano.tensor as T

import numpy as np

from layers import DataLayer, ConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer


class AlexNet(object):

    def __init__(self, config):

        self.config = config
        batch_size = config['batch_size']
        flag_datalayer = config['use_data_layer']
        lib_conv = config['lib_conv']

        ################## BUILD NETWORK ############
        # allocate the symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        x = T.ftensor4('x')
        y = T.ivector('y')

        rand = T.fvector('rand')

        print '... building the model'
        self.layers = []
        params = []
        weight_types = []

        if flag_datalayer:
            data_layer = DataLayer(input=x,
                                   image_shape=(batch_size, 3, 256, 256),
                                   cropsize=227,
                                   rand=rand,
                                   mirror=True,
                                   flag_rand=config['rand_crop']
                                   )

            layer1_input = data_layer.output
        else:
            layer1_input = x

        convpool_layer1 = ConvPoolLayer(input=layer1_input,
                                        input_shape=(batch_size, 3, 227, 227),
                                        filter_shape=(96, 3, 11, 11),
                                        conv_stride=4,
                                        padding=0,
                                        group=1,
                                        pool_size=(3, 3),
                                        pool_stride=2,
                                        bias_init=0.0
                                        )

        self.layers.append(convpool_layer1)
        params += convpool_layer1.params
        weight_types += convpool_layer1.weight_type

        convpool_layer2 = ConvPoolLayer(input=convpool_layer1.output,
                                        input_shape=(batch_size, 96, 27, 27),
                                        filter_shape=(256, 96, 5, 5),
                                        conv_stride=1,
                                        padding=2,
                                        group=1,
                                        pool_size=(3, 3),
                                        pool_stride=2,
                                        bias_init=0.1
                                        )

        self.layers.append(convpool_layer2)
        params += convpool_layer2.params
        weight_types += convpool_layer2.weight_type

        convpool_layer3 = ConvPoolLayer(input=convpool_layer2.output,
                                        input_shape=(batch_size, 256, 13, 13),
                                        filter_shape=(384, 256, 3, 3),
                                        conv_stride=1,
                                        padding=1,
                                        group=1,
                                        pool_size=(1, 1),
                                        pool_stride=0,
                                        bias_init=0.0
                                        )

        self.layers.append(convpool_layer3)
        params += convpool_layer3.params
        weight_types += convpool_layer3.weight_type

        convpool_layer4 = ConvPoolLayer(input=convpool_layer3.output,
                                        input_shape=(batch_size, 384, 13, 13),
                                        filter_shape=(384, 384, 3, 3),
                                        conv_stride=1,
                                        padding=1,
                                        group=1,
                                        pool_size=(1, 1),
                                        pool_stride=0,
                                        bias_init=0.1
                                        )

        self.layers.append(convpool_layer4)
        params += convpool_layer4.params
        weight_types += convpool_layer4.weight_type

        convpool_layer5 = ConvPoolLayer(input=convpool_layer4.output,
                                        input_shape=(batch_size, 384, 13, 13),
                                        filter_shape=(256, 384, 3, 3),
                                        conv_stride=1,
                                        padding=1,
                                        group=1,
                                        pool_size=(3, 3),
                                        pool_stride=2,
                                        bias_init=0.0
                                        )

        self.layers.append(convpool_layer5)
        params += convpool_layer5.params
        weight_types += convpool_layer5.weight_type

        fc_layer6_input = T.flatten(convpool_layer5.output, outdim=2)

        fc_layer6 = FCLayer(input=fc_layer6_input,
                            n_in=256 * 6 * 6,
                            n_out=4096
                            )

        self.layers.append(fc_layer6)
        params += fc_layer6.params
        weight_types += fc_layer6.weight_type

        dropout_layer6 = DropoutLayer(input=fc_layer6.output,
                                      n_in=4096,
                                      n_out=4096
                                      )

        fc_layer7 = FCLayer(input=dropout_layer6.output,
                            n_in=4096,
                            n_out=4096
                            )

        self.layers.append(fc_layer7)
        params += fc_layer7.params
        weight_types += fc_layer7.weight_type

        dropout_layer7 = DropoutLayer(input=fc_layer7.output,
                                      n_in=4096,
                                      n_out=4096
                                      )

        softmax_layer8 = SoftmaxLayer(input=dropout_layer7.output,
                                      n_in=4096,
                                      n_out=1000
                                      )

        self.layers.append(softmax_layer8)
        params += softmax_layer8.params
        weight_types += softmax_layer8.weight_type

        # #################### NETWORK BUILT ####################
        self.cost = softmax_layer8.negetive_log_likelihood(y)
        self.errors = softmax_layer8.errors(y)
        self.errors_top_5 = softmax_layer8.errors_top_x(y, 5)
        self.params = params
        self.x = x
        self.y = y
        self.rand = rand
        self.weight_types = weight_types
        self.batch_size = batch_size


def compile_models(model, config, flag_top_5=False):
    '''
    '''
    x = model.x
    y = model.y
    rand = model.rand
    weight_types = model.weight_types

    cost = model.cost
    params = model.params
    errors = model.errors
    errors_top_5 = model.errors_top_5
    batch_size = model.batch_size

    mu = config['momentum']
    eta = config['weight_decay']

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = []

    learning_rate = theano.shared(np.float32(config['learning_rate']))

    lr = T.scalar('lr')  # symbolic learning rate

    if config['use_data_layer']:
        raw_size = 256
    else:
        raw_size = 227

    shared_x = theano.shared(np.zeros((batch_size, 3, raw_size, raw_size),
                                      dtype=theano.config.floatX
                                      ),
                             borrow=True
                             )

    shared_y = theano.shared(np.zeros((batch_size,),
                                      dtype=int
                                      ),
                             borrow=True
                             )

    rand_arr = theano.shared(np.zeros(3,
                                      dtype=theano.config.floatX
                                      ),
                             borrow=True
                             )

    vels = [theano.shared(param_i.get_value() * 0.0) for param_i in params]

    if config['use_momentum']:

        assert len(weight_types) == len(params)

        for param_i, grad_i, vel_i, weight_type in zip(params, grads, vels, weight_types):

            if weight_type == 'W':

                real_grad = grad_i + eta * param_i
                real_lr = lr

            elif weight_type == 'b':

                real_grad = grad_i
                real_lr = 2.0 * lr

            else:
                raise TypeError('Weight Type Error')

            if config['use_nesterov_momentum']:

                vel_i_next = mu ** 2 * vel_i - (1 + mu) * real_lr * real_grad

            else:

                vel_i_next = mu * vel_i - real_lr * real_grad

            updates.append((vel_i, vel_i_next))
            updates.append((param_i, param_i + vel_i_next))

    else:
        for param_i, grad_i, weight_type in zip(params, grads, weight_types):

            if weight_type == 'W':

                updates.append((param_i, param_i - lr * grad_i - eta * lr * param_i))

            elif weight_type == 'b':

                updates.append((param_i, param_i - 2.0 * lr * grad_i))

            else:
                raise TypeError('Weight Type Error')

    # Define Theano Function
    train_model = theano.function(
        inputs=[],
        outputs=cost,
        updates=updates,
        givens=[
            (x, shared_x),
            (y, shared_y),
            (lr, learning_rate),
            (rand, rand_arr)
        ],
        name='train_model'
    )

    validate_outputs = [cost, errors]

    if flag_top_5:
        validate_outputs.append(errors_top_5)

    validate_model = theano.function(
        inputs=[],
        outputs=validate_outputs,
        givens=[
            (x, shared_x),
            (y, shared_y),
            (rand, rand_arr)
        ],
        name='validate_model'
    )

    test_model = theano.function(
        inputs=[],
        outputs=errors,
        givens=[
            (x, shared_x),
            (y, shared_y),
            (rand, rand_arr)
        ],
        name='test_model'
    )

    return (train_model, validate_model, test_model, learning_rate, shared_x, shared_y, rand_arr, vels)
