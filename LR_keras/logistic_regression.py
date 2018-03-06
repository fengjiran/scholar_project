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

from load_data import load_mnist


def train_logistic_regression(config):
    '''
    '''
    n_in = config['n_in']
    n_out = config['n_out']
    activation = config['activation']
    init = config['weights_init']
    learning_rate = config['learning_rate']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    loss = config['loss']

    (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y) = load_mnist()
    train_set_y = to_categorical(train_set_y, 10)
    valid_set_y = to_categorical(valid_set_y, 10)
    test_set_y = to_categorical(test_set_y, 10)

    # initialize the weights
    rng = np.random.RandomState(1234)
    W = np.asarray(rng.normal(loc=0,
                              scale=0.01,
                              size=(n_in, n_out)
                              ),
                   dtype=keras.backend.floatx()
                   )

    b = np.zeros((n_out,), dtype=keras.backend.floatx())

    weights = [W, b]

    model = Sequential(name='LogisticRegression')
    model.add(Dense(output_dim=n_out,
                    # weights=weights,
                    init=init,
                    input_dim=n_in,
                    activation=activation,
                    name='FC_0'
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

    # output = model.name # get the model name if exists, 'LogisticRegression'
    # output = model.get_layer(index=1).name  # get the layer name, FC_0
    # output = model.layers[0].name # FC_0

    # intermediate_layer_model = Model(input=model.input,
    #                                  output=model.get_layer('FC_0').output
    #                                  )

    # intermediate_output = intermediate_layer_model.predict(test_set_x[0].reshape(1, 784))
    # print intermediate_output


if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    train_logistic_regression(config)
