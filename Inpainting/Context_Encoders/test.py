from __future__ import division
import os
from glob import glob
import platform
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU
from keras.layers.core import Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.regularizers import l2
from keras import backend as K

from utils import load_image, crop_random

batch_size = 500
weight_decay_rate = 0.0001

if platform.system() == 'Windows':
    trainset_path = 'X:\\DeepLearning\\imagenet_trainset.pickle'
    testset_path = 'X:\\DeepLearning\\imagenet_testset.pickle'
    dataset_path = 'X:\\DeepLearning\\ImageNet_100K'

elif platform.system() == 'Linux':
    trainset_path = '/home/richard/Deep_Learning_Projects/Inpainting/Context_Encoders/imagenet_trainset.pickle'
    testset_path = '/home/richard/Deep_Learning_Projects/Inpainting/Context_Encoders/imagenet_testset.pickle'
    dataset_path = '/home/richard/datasets/ImageNet_100K'


if not os.path.exists(trainset_path) or not os.path.exists(testset_path):
    imagenet_images = []
    for filepath, _, _ in os.walk(dataset_path):
        imagenet_images.extend(glob(os.path.join(filepath, '*.JPEG')))

    imagenet_images = np.hstack(imagenet_images)

    trainset = pd.DataFrame({'image_path': imagenet_images[:int(len(imagenet_images) * 0.9)]})
    testset = pd.DataFrame({'image_path': imagenet_images[int(len(imagenet_images) * 0.9):]})

    trainset.to_pickle(trainset_path)
    testset.to_pickle(testset_path)

else:
    trainset = pd.read_pickle(trainset_path)
    testset = pd.read_pickle(testset_path)

# trainset.index = range(len(trainset))
# trainset = trainset.ix[np.random.permutation(len(trainset))]

# for start, end in zip(range(0, len(trainset), batch_size), range(batch_size, len(trainset), batch_size)):
#     image_paths = trainset[start:end]['image_path'].values
#     images_ori = map(load_image, image_paths)
#     is_none = np.sum([x is None for x in images_ori])
#     # is_none = np.sum(map(lambda x: x is None, images_ori))
#     if is_none > 0:
#         continue

#     images_crops = map(crop_random, images_ori)
#     images, crops, _, _ = zip(*images_crops)

# testset.index = range(len(testset))
# testset = testset.ix[np.random.permutation(len(testset))]

# for start, end in zip(range(0, len(testset), batch_size), range(batch_size, len(testset), batch_size)):
#     image_paths = testset[start:end]['image_path'].values
#     images_ori = map(load_image, image_paths)
#     is_none = np.sum([x is None for x in images_ori])
#     # is_none = np.sum(map(lambda x: x is None, images_ori))
#     if is_none > 0:
#         continue

#     images_crops = map(crop_random, images_ori)
#     images, crops, _, _ = zip(*images_crops)

inpt = Input(shape=(3, 128, 128))
x = ZeroPadding2D(padding=(1, 1))(inpt)
conv1 = Conv2D(filters=64,
               kernel_size=(3, 3),
               strides=(2, 2),
               kernel_regularizer=l2(weight_decay_rate),
               name='conv1')(x)

bn1 = BatchNormalization(axis=1)(conv1)
x = Activation(LeakyReLU(0.1))(bn1)
x = ZeroPadding2D(padding=(1, 1))(x)
conv2 = Conv2D(filters=64,
               kernel_size=(3, 3),
               strides=(2, 2),
               kernel_regularizer=l2(weight_decay_rate),
               name='conv2')(x)

bn2 = BatchNormalization(axis=1)(conv2)
x = Activation(LeakyReLU(0.1))(bn2)
x = ZeroPadding2D(padding=(1, 1))(x)
conv3 = Conv2D(filters=128,
               kernel_size=(3, 3),
               strides=(2, 2),
               kernel_regularizer=l2(weight_decay_rate),
               name='conv3')(x)

bn3 = BatchNormalization(axis=1)(conv3)
x = Activation(LeakyReLU(0.1))(bn3)

x = ZeroPadding2D(padding=(1, 1))(x)
conv4 = Conv2D(filters=256,
               kernel_size=(3, 3),
               strides=(2, 2),
               kernel_regularizer=l2(weight_decay_rate),
               name='conv4')(x)

bn4 = BatchNormalization(axis=1)(conv4)
x = Activation(LeakyReLU(0.1))(bn4)

x = ZeroPadding2D(padding=(1, 1))(x)
conv5 = Conv2D(filters=512,
               kernel_size=(3, 3),
               strides=(2, 2),
               kernel_regularizer=l2(weight_decay_rate),
               name='conv5')(x)

bn5 = BatchNormalization(axis=1)(conv5)
x = Activation(LeakyReLU(0.1))(bn5)

conv6 = Conv2D(filters=4000,
               kernel_size=(3, 3),
               strides=(2, 2),
               kernel_regularizer=l2(weight_decay_rate),
               name='conv6')(x)

bn6 = BatchNormalization(axis=1)(conv6)
x = Activation(LeakyReLU(0.1))(bn6)
x = UpSampling2D(size=(2, 2))(x)
deconv4 = Conv2DTranspose(filters=512,
                          #   strides=(2, 2),
                          kernel_size=(3, 3),
                          data_format=K.image_data_format())(x)

debn4 = BatchNormalization(axis=1)(deconv4)
x = Activation('relu')(debn4)

x = UpSampling2D(size=(2, 2))(x)
deconv3 = Conv2DTranspose(filters=256,
                          #   strides=(2, 2),
                          kernel_size=(3, 3),
                          padding='same',
                          kernel_regularizer=l2(weight_decay_rate),
                          data_format=K.image_data_format())(x)

debn3 = BatchNormalization(axis=1)(deconv3)
x = Activation('relu')(debn3)

x = UpSampling2D(size=(2, 2))(x)
deconv2 = Conv2DTranspose(filters=128,
                          #   strides=(2, 2),
                          kernel_size=(3, 3),
                          padding='same',
                          kernel_regularizer=l2(weight_decay_rate),
                          data_format=K.image_data_format())(x)
debn2 = BatchNormalization()(deconv2)
x = Activation('relu')(debn2)

x = UpSampling2D(size=(2, 2))(x)
deconv1 = Conv2DTranspose(filters=64,
                          #   strides=(2, 2),
                          kernel_size=(3, 3),
                          padding='same',
                          kernel_regularizer=l2(weight_decay_rate),
                          data_format=K.image_data_format())(x)
debn1 = BatchNormalization()(deconv1)
x = Activation('relu')(debn1)

x = UpSampling2D(size=(2, 2))(x)
x = Conv2DTranspose(filters=3,
                    #  strides=(2, 2),
                    kernel_size=(3, 3),
                    padding='same',
                    kernel_regularizer=l2(weight_decay_rate),
                    data_format=K.image_data_format())(x)

model = Model(inputs=inpt, outputs=[x, debn1])

print model.summary()
a = np.ones((100, 3, 128, 128))
target1 = np.ones((100, 3, 64, 64))
target2 = np.ones((100, 64, 32, 32))
model.compile(loss=['mse', 'mse'],
              loss_weights=[1, -1],
              optimizer='sgd')

loss = model.train_on_batch([a], [target1, target2])
print loss

# print 'Iter: {0},\
#        Recon loss: {1},\
#        Gen adv loss: {2},\
#        Gen loss: {3},\
#        Dis loss: {4}'.format(0,
#                              1,
#                              2,
#                              3,
#                              4)

# b = model.predict(a)

# print b.shape
