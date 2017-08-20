from __future__ import division

import os
import sys
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.core import Flatten
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
from keras import backend as K

import csv
import yaml
import h5py
import math
import numpy as np

from load_cifar10 import load_cifar_10
from load_cifar10 import global_contrast_normalize
from load_cifar10 import zca


class TestCallback(Callback):
    '''
    ref: https://github.com/fchollet/keras/issues/2548
    '''

    def __init__(self, test_data, test_history_filepath):
        self.test_data = test_data
        self.test_history_filepath = test_history_filepath

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, batch_size=250, verbose=0)

        with open(self.test_history_filepath, 'a+') as f:
            mywrite = csv.writer(f)
            if epoch == 0:
                mywrite.writerow(['test_loss', 'test_acc'])
                mywrite.writerow([loss, acc])
            else:
                mywrite.writerow([loss, acc])

        print '\nTesting loss: {0}, acc: {1}'.format(loss, acc)


class LrDecay(Callback):

    def __init__(self, initial_lr, e1, e2, e3, drop_rate):
        self.initial_lr = initial_lr
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.drop_rate = drop_rate

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.e1:
            K.set_value(self.model.optimizer.lr, self.initial_lr * self.drop_rate)
        if epoch == self.e2:
            K.set_value(self.model.optimizer.lr, self.initial_lr * (self.drop_rate**2))
        if epoch == self.e3:
            K.set_value(self.model.optimizer.lr, self.initial_lr * (self.drop_rate**3))

        print '\nThe learning rate is: {:.6f}\n'.format(K.eval(self.model.optimizer.lr))


class model_c(object):

    def __init__(self, activation='relu', weight_decay=0.001):
        '''
        Construct the network
        '''
        self.activation = activation
        self.weight_decay = weight_decay

        self.model = Sequential()
        self.model.add(Dropout(0.2, input_shape=(3, 32, 32)))

        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(filters=96,
                              kernel_size=(3, 3),
                              activation=self.activation,
                              kernel_regularizer=l2(self.weight_decay)))

        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(filters=96,
                              kernel_size=(3, 3),
                              activation=self.activation,
                              kernel_regularizer=l2(self.weight_decay)))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        self.model.add(Dropout(0.5))

        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(filters=192,
                              kernel_size=(3, 3),
                              activation=self.activation,
                              kernel_regularizer=l2(self.weight_decay)))

        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(filters=192,
                              kernel_size=(3, 3),
                              activation=self.activation,
                              kernel_regularizer=l2(self.weight_decay)))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        self.model.add(Dropout(0.5))

        self.model.add(ZeroPadding2D(padding=(1, 1)))
        self.model.add(Conv2D(filters=192,
                              kernel_size=(3, 3),
                              activation=self.activation,
                              kernel_regularizer=l2(self.weight_decay)))
        self.model.add(Conv2D(filters=192,
                              kernel_size=(1, 1),
                              activation=self.activation,
                              kernel_regularizer=l2(self.weight_decay)))
        self.model.add(Conv2D(filters=10,
                              kernel_size=(1, 1),
                              activation=self.activation,
                              kernel_regularizer=l2(self.weight_decay)))

        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(units=10,
                             activation='softmax',
                             kernel_regularizer=l2(self.weight_decay)))

        print self.model.summary()

    def train(self,
              initial_lr=0.05,
              momentum=0.9,
              lr_decay=0,
              batch_size=250,
              train_epochs=350,
              lr_scheduler=True,
              e1=100,
              e2=200,
              e3=300,
              drop_rate=0.1,
              train_history_filepath='train_history_c.csv',
              test_history_filepath='test_history_c.csv'
              ):
        self.initial_lr = initial_lr
        # self.drop_rate = drop_rate
        # self.epochs_drop = epochs_drop
        # self.batch_size = batch_size
        # self.train_epochs = train_epochs

        (X_train, y_train), (X_test, y_test) = load_cifar_10()

        # preprocess
        X_train = np.reshape(X_train, (X_train.shape[0], 3 * 32 * 32))
        X_test = np.reshape(X_test, (X_test.shape[0], 3 * 32 * 32))

        # gcn and zca
        X_train = global_contrast_normalize(X_train)
        X_test = global_contrast_normalize(X_test)

        X_train = zca(X_train)
        X_test = zca(X_test)

        X_train = np.reshape(X_train, (X_train.shape[0], 3, 32, 32))
        X_test = np.reshape(X_test, (X_test.shape[0], 3, 32, 32))

        if lr_scheduler:
            sgd = SGD(lr=initial_lr, momentum=momentum, nesterov=True)
            self.model.compile(optimizer=sgd,
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

            history = self.model.fit(X_train, y_train,
                                     epochs=train_epochs,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     callbacks=[TestCallback((X_test, y_test), test_history_filepath),
                                                LrDecay(initial_lr, e1, e2, e3, drop_rate)
                                                ])
            return history
        else:
            sgd = SGD(lr=initial_lr, decay=lr_decay, momentum=momentum, nesterov=True)
            self.model.compile(optimizer=sgd,
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

            print 'The total params: {}'.format(self.num_params())

            self.model.fit(X_train, y_train,
                           epochs=train_epochs,
                           batch_size=batch_size,
                           shuffle=True,
                           callbacks=[TestCallback((X_test, y_test), test_history_filepath)])
            return history

    def get_config(self):
        '''
        Returns a dictionary containing the configuration of the model.
        The model can be reinstantiated from its config via:
        config = model.get_config()
        model = Model.from_config(config)
        or for Sequential:
        model = Sequential.from_config(config)
        '''
        self.config = self.model.get_config()
        return self.config

    def save_model_weights(self, filepath='model_c_weights.h5'):

        self.model.save_weights(filepath)

    def num_params(self):
        '''
        Count the number of parameters in the network
        '''
        return self.model.count_params()

if __name__ == '__main__':
    # train()
    with open('config_c.yaml', 'r') as f:
        config = yaml.load(f)

    activation = config['activation']
    weight_decay = config['weight_decay']
    initial_lr = config['initial_lr']
    momentum = config['momentum']
    lr_decay = config['lr_decay']
    batch_size = config['batch_size']
    train_epochs = config['train_epochs']
    lr_scheduler = config['lr_scheduler']

    e1 = config['e1']
    e2 = config['e2']
    e3 = config['e3']

    drop_rate = config['drop_rate']

    train_history_filepath = config['train_history_filepath']
    test_history_filepath = config['test_history_filepath']

    model = model_c(activation=activation, weight_decay=weight_decay)

    start_time = time.clock()

    hist = model.train(initial_lr=initial_lr,
                       momentum=momentum,
                       lr_decay=lr_decay,
                       batch_size=batch_size,
                       train_epochs=train_epochs,
                       lr_scheduler=lr_scheduler,
                       e1=e1,
                       e2=e2,
                       e3=e3,
                       drop_rate=drop_rate,
                       train_history_filepath=train_history_filepath,
                       test_history_filepath=test_history_filepath
                       )

    model.save_model_weights()

    train_loss = hist.history['loss']
    train_acc = hist.history['acc']

    with open(train_history_filepath, 'a+') as f:
        mywrite = csv.writer(f)
        mywrite.writerow(['train_loss', 'train_acc'])
        for loss, acc in zip(train_loss, train_acc):
            mywrite.writerow([loss, acc])

    end_time = time.clock()

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
