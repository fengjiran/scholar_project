from __future__ import division

import numpy as np
import yaml

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2

from load_data import load_mnist


def train_mlp(config):
    '''
    '''
    n_in = config['n_in']
    n_hidden = config['n_hidden']
    n_out = config['n_out']
    # activation = config['activation']
    init = config['weights_init']
    learning_rate = config['learning_rate']
    l2_reg = config['l2_reg']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    loss = config['loss']

    (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y) = load_mnist()
    train_set_y = to_categorical(train_set_y)
    valid_set_y = to_categorical(valid_set_y)
    test_set_y = to_categorical(test_set_y)

    model = Sequential(name='MLP')
    model.add(Dense(output_dim=n_hidden,
                    input_dim=n_in,
                    W_regularizer=l2(l2_reg),
                    init='glorot_uniform',
                    activation='tanh',
                    name='FC_0'
                    )
              )

    model.add(Dense(output_dim=n_out,
                    W_regularizer=l2(l2_reg),
                    init='zero',
                    activation='softmax',
                    name='Softmax'
                    )
              )

    model.compile(optimizer=SGD(lr=learning_rate),
                  loss=loss,
                  metrics=['accuracy']
                  )

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5000
                                   )

    model.fit(train_set_x, train_set_y,
              batch_size=batch_size,
              nb_epoch=n_epochs,
              validation_data=(valid_set_x, valid_set_y),
              callbacks=[early_stopping]
              )

    score = model.evaluate(test_set_x, test_set_y, batch_size=batch_size)
    print score


if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    train_mlp(config)
