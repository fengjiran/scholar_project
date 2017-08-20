from __future__ import division

import numpy as np
import yaml

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2
from keras import backend as K

from load_data import load_mnist

batch_size = 128
nb_classes = 10
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 28, 28

# number of convolutional filters to use
nb_filters = 32

# size of pooling area for max pooling
pool_size = (2, 2)

# convolution kernel size
kernel_size = (3, 3)

(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y) = load_mnist()
train_set_x = np.concatenate([train_set_x, valid_set_x], axis=0)
train_set_y = np.concatenate([train_set_y, valid_set_y])

if K.image_dim_ordering() == 'th':
    train_set_x = train_set_x.reshape(train_set_x.shape[0], 1, img_rows, img_cols)
    test_set_x = test_set_x.reshape(test_set_x.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_set_x = train_set_x.reshape(train_set_x.shape[0], img_rows, img_cols, 1)
    test_set_x = test_set_x.reshape(test_set_x.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

train_set_x = train_set_x.astype('float32')
test_set_x = test_set_x.astype('float32')

train_set_x /= 255
test_set_x /= 255

print 'train_set_x shape: {}'.format(train_set_x.shape)
print '{} train samples'.format(train_set_x.shape[0])
print '{} test samples'.format(test_set_x.shape[0])

train_set_y = to_categorical(train_set_y)
test_set_y = to_categorical(test_set_y)

model = Sequential(name='LeNet-5')

model.add(Convolution2D(nb_filter=nb_filters,
                        nb_row=kernel_size[0],
                        nb_col=kernel_size[1],
                        activation='relu',
                        border_mode='valid',
                        input_shape=input_shape,
                        name='conv1'
                        )
          )

model.add(Convolution2D(nb_filter=nb_filters,
                        nb_row=kernel_size[0],
                        nb_col=kernel_size[1]
                        )
          )

model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(output_dim=128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(output_dim=nb_classes, activation='softmax'))

model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(train_set_x, train_set_y,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          verbose=1,
          validation_data=(test_set_x, test_set_y)
          )

score = model.evaluate(test_set_x, test_set_y)

print 'Test score: {}'.format(score[0])
print 'Test accuracy: {}'.format(score[1])
