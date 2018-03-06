from __future__ import division

import cPickle

# import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import SGD

from densenet import DenseNet
from load_data import load_svhn

n_epochs = 40
batch_size = 64
depth_40 = 40
depth_100 = 100
depth_190 = 190
depth_250 = 250

nb_dense_block = 3
nb_filter = 16
growth_rate_12 = 12
growth_rate_24 = 24
growth_rate_40 = 40
dropout_rate = 0.2
init_lr = 0.1
weight_decay = 1e-4


class TestCallback(Callback):
    """ref: https://github.com/fchollet/keras/issues/2548."""

    def __init__(self, test_data):
        super(TestCallback, self).__init__()
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):

        self.model.save_weights('./model_svhn/model1.h5')

        with open('./model_svhn/start_epoch', 'wb') as f:
            cPickle.dump(epoch, f, True)

        with open('./model_svhn/current_lr', 'wb') as f:
            cPickle.dump(K.get_value(self.model.optimizer.lr), f, True)

        if epoch % 10 == 0:
            x, y = self.test_data
            loss, acc = self.model.evaluate(x, y, batch_size=64, verbose=0)
            print '\nTesting loss: {0}, acc: {1}'.format(loss, acc)


class LrDecay(Callback):
    """The class of lr decay."""

    def __init__(self, init_lr, e1, e2):
        super(LrDecay, self).__init__()
        self.init_lr = init_lr
        self.e1 = e1
        self.e2 = e2

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.e1:
            K.set_value(self.model.optimizer.lr, K.get_value(self.model.optimizer.lr) * 0.1)
        if epoch == self.e2:
            K.set_value(self.model.optimizer.lr, K.get_value(self.model.optimizer.lr) * 0.1)

        print '\nThe learning rate is: {:.6f}\n'.format(K.eval(self.model.optimizer.lr))


###################
# Data processing #
###################
isFirstTimeTrain = False

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = load_svhn()

img_dim = X_train.shape[1:]  # (3, 32, 32)

if K.image_data_format() == 'channels_first':
    n_channels = X_train.shape[1]
else:
    n_channels = X_train.shape[-1]

X_val = X_train[0:6000]
y_val = y_train[0:6000]

X_train = X_train[6000:]
y_train = y_train[6000:]

model1 = DenseNet(nb_classes=10,
                  img_dim=img_dim,
                  depth=depth_40,
                  nb_dense_block=nb_dense_block,
                  growth_rate=growth_rate_12,
                  nb_filter=nb_filter,
                  dropout_rate=dropout_rate,
                  weight_decay=weight_decay)

opt = SGD(lr=init_lr, momentum=0.9)
model1.compile(loss='categorical_crossentropy',
               optimizer=opt,
               metrics=['accuracy'])

if not isFirstTimeTrain:
    model1.load_weights('./model_svhn/model1.h5')

    with open('./model_svhn/start_epoch') as f:
        start_epoch = cPickle.load(f)

    with open('./model_svhn/current_lr') as f:
        current_lr = cPickle.load(f)

    K.set_value(model1.optimizer.lr, current_lr)

else:
    start_epoch = 0  # from the epoch index of 0

    model1.save_weights('./model_svhn/model1.h5')

    with open('./model_svhn/start_epoch', 'wb') as f:
        cPickle.dump(start_epoch, f, True)

    with open('./model_svhn/current_lr', 'wb') as f:
        cPickle.dump(K.get_value(model1.optimizer.lr), f, True)

results = model1.fit(x=X_train,
                     y=y_train,
                     validation_data=(X_val, y_val),
                     batch_size=batch_size,
                     epochs=n_epochs,
                     initial_epoch=start_epoch,
                     callbacks=[TestCallback((X_test, y_test)),
                                LrDecay(init_lr, 15, int(n_epochs * 0.75))])
