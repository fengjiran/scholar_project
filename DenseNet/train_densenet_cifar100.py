from __future__ import division

import cPickle

import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import SGD

from densenet import DenseNet
from load_data import load_cifar_100

n_epochs = 300
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

        self.model.save_weights('./model_cifar100/model1.h5')

        with open('./model_cifar100/start_epoch', 'wb') as f:
            cPickle.dump(epoch, f, True)

        with open('./model_cifar100/current_lr', 'wb') as f:
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
isFirstTimeTrain = True
data_augmentation = False

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = load_cifar_100()

n_train_samples = X_train.shape[0]
n_test_samples = X_test.shape[0]

n_train_batches = int(n_train_samples / batch_size)
n_test_batches = int(n_test_samples / batch_size)

# nb_classes = len(np.unique(y_train))
img_dim = X_train.shape[1:]  # (3, 32, 32)

if K.image_data_format() == 'channels_first':
    n_channels = X_train.shape[1]
else:
    n_channels = X_train.shape[-1]

if data_augmentation:
    # 4 zero padding
    X_train = np.pad(X_train, pad_width=((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')

    # crop to (32,32)
    random_y = np.random.randint(0, 8)
    random_x = np.random.randint(0, 8)
    X_train = X_train[:, :, random_y:random_y + 32, random_x:random_x + 32]

    X1 = X_train[0:25000, :, ::-1, :]
    X2 = X_train[25000:]

    X = np.vstack((X1, X2, X_test))

    if K.image_data_format() == 'channels_first':
        for i in range(n_channels):
            mean = np.mean(X[:, i, :, :])
            std = np.std(X[:, i, :, :])
            X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
            X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std
    else:
        for i in range(n_channels):
            mean = np.mean(X[:, :, :, i])
            std = np.std(X[:, :, :, i])
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

    dropout_rate = None

else:
    X = np.vstack((X_train, X_test))

    if K.image_data_format() == 'channels_first':
        for i in range(n_channels):
            mean = np.mean(X[:, i, :, :])
            std = np.std(X[:, i, :, :])
            X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
            X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std
    else:
        for i in range(n_channels):
            mean = np.mean(X[:, :, :, i])
            std = np.std(X[:, :, :, i])
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std


model1 = DenseNet(nb_classes=100,
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
    model1.load_weights('./model_cifar100/model1.h5')

    with open('./model_cifar100/start_epoch') as f:
        start_epoch = cPickle.load(f)

    with open('./model_cifar100/current_lr') as f:
        current_lr = cPickle.load(f)

    K.set_value(model1.optimizer.lr, current_lr)

else:
    start_epoch = 0  # from the epoch index of 0

    model1.save_weights('./model_cifar100/model1.h5')

    with open('./model_cifar100/start_epoch', 'wb') as f:
        cPickle.dump(start_epoch, f, True)

    with open('./model_cifar100/current_lr', 'wb') as f:
        cPickle.dump(K.get_value(model1.optimizer.lr), f, True)


results = model1.fit(x=X_train,
                     y=y_train,
                     batch_size=batch_size,
                     epochs=n_epochs,
                     initial_epoch=start_epoch,
                     callbacks=[TestCallback((X_test, y_test)),
                                LrDecay(init_lr, int(n_epochs * 0.1), int(n_epochs * 0.2))])
