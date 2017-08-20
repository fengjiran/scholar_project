from __future__ import division

import cPickle
import os
import platform
import sys

import numpy as np
from keras import backend as K
# from keras import callbacks
from keras import metrics
from keras.optimizers import SGD
from keras.utils import to_categorical

from googlenet_v1 import GoogLeNet_v1
from imagenet_utils import preprocess_input, read_batch, read_validation_batch

if platform.system() == 'Windows':
    train_path = 'X:\\DeepLearning\\ILSVRC2015\\ILSVRC2015_CLS-LOC\\ILSVRC2015_CLS-LOC\\ILSVRC2015\\Data\\CLS-LOC\\train'
    val_path = 'X:\\DeepLearning\\ILSVRC2015\\ILSVRC2015_CLS-LOC\\ILSVRC2015_CLS-LOC\\ILSVRC2015\\Data\\CLS-LOC\\val'
    test_path = 'X:\\DeepLearning\\ILSVRC2015\\ILSVRC2015_CLS-LOC\\ILSVRC2015_CLS-LOC\\ILSVRC2015\\Data\\CLS-LOC\\test'
    val_annotation_path = 'X:\\DeepLearning\\ILSVRC2015\\ILSVRC2015_CLS-LOC\\ILSVRC2015_CLS-LOC\\ILSVRC2015\\Annotations\\CLS-LOC\\val'
elif platform.system() == 'Linux':
    train_path = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/train'
    val_path = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/val'
    val_annotation_path = '/media/icie/b29b7268-50ad-4752-8e03-457669cab10a/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Annotations/CLS-LOC/val'

    # train_path = '/mnt/DataBase/DeepLearning/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/train'
    # val_path = '/mnt/DataBase/DeepLearning/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/val'
    # val_annotation_path = '/mnt/DataBase/DeepLearning/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Annotations/CLS-LOC/val'

n_train_samples = 1281166
n_val_samples = 50000
n_class = 1000
n_epochs = 100
train_batch_size = 128
val_batch_size = 128
init_lr = 0.01
# test_step = 20000

isFirstTimeTrain = False

n_train_batches = int(n_train_samples / train_batch_size)
n_val_batches = int(n_val_samples / val_batch_size)

model = GoogLeNet_v1()

opt = SGD(lr=init_lr, momentum=0.9)

model.compile(loss=['categorical_crossentropy',
                    'categorical_crossentropy',
                    'categorical_crossentropy'],
              optimizer=opt,
              loss_weights=[0.3, 0.3, 1.],
              metrics=['accuracy', metrics.top_k_categorical_accuracy])

print model.metrics_names

if not isFirstTimeTrain:
    model.load_weights('model.h5')

    with open('start_epoch') as f:
        start_epoch = cPickle.load(f)

    with open('start_iter') as f:
        start_iter = cPickle.load(f)

    with open('current_lr') as f:
        current_lr = cPickle.load(f)

    K.set_value(model.optimizer.lr, current_lr)

else:
    start_epoch = 0
    start_iter = 0
    current_lr = K.get_value(model.optimizer.lr)

    model.save_weights('model.h5')

    with open('start_epoch', 'wb') as f:
        cPickle.dump(start_epoch, f, True)

    with open('start_iter', 'wb') as f:
        cPickle.dump(start_iter, f, True)

    with open('current_lr', 'wb') as f:
        cPickle.dump(current_lr, f, True)


for epoch in range(start_epoch, n_epochs):
    print 'Epoch: {0}/{1}'.format(epoch + 1, n_epochs)

    for iters in range(start_iter, n_train_batches):
        print '============================================'
        print 'Iter: {0}/{1}'.format(iters, n_train_batches)

        images, labels = read_batch(train_batch_size, train_path)
        images = preprocess_input(images)
        labels = to_categorical(labels, n_class)

        results = model.train_on_batch(x=images,
                                       y=[labels, labels, labels])

        print 'Loss: {}'.format(results[0])
        print 'branch 1 loss: {}'.format(results[1])
        print 'branch 2 loss: {}'.format(results[2])
        print 'branch 3 loss: {}'.format(results[3])

        print 'branch 1 top-1 acc: {}'.format(results[4])
        print 'branch 1 top-5 acc: {}'.format(results[5])

        print 'branch 2 top-1 acc: {}'.format(results[6])
        print 'branch 2 top-5 acc: {}'.format(results[7])

        print 'branch 3 top-1 acc: {}'.format(results[8])
        print 'branch 3 top-5 acc: {}'.format(results[9])
        print '============================================='
        print '\n'

        # results = model.fit(x=images,
        #                     y=[labels, labels, labels],
        #                     batch_size=train_batch_size,
        #                     epochs=1)

        if (iters != 0) and (iters % 500 == 0):
            model.save_weights('model.h5')

            with open('start_epoch', 'wb') as f:
                cPickle.dump(epoch, f, True)

            with open('start_iter', 'wb') as f:
                cPickle.dump(iters, f, True)

            with open('current_lr', 'wb') as f:
                cPickle.dump(K.get_value(model.optimizer.lr), f, True)

    with open('start_iter', 'wb') as f:
        cPickle.dump(0, f, True)

    print 'Evaulate the model...'
    val_acc = []
    val_loss = []
    for i in range(n_val_batches):
        print 'Evaluate Iter: {0}/{1}'.format(i, n_val_batches)
        images, labels = read_validation_batch(batch_size=val_batch_size,
                                               val_path=val_path,
                                               val_annotation_path=val_annotation_path)

        images = preprocess_input(images)
        labels = to_categorical(labels, n_class)

        results = model.test_on_batch(x=images,
                                      y=[labels, labels, labels])

        print 'Eva loss: {}'.format(results[0])
        print 'Eva branch 1 loss: {}'.format(results[1])
        print 'Eva branch 2 loss: {}'.format(results[2])
        print 'Eva branch 3 loss: {}'.format(results[3])

        print 'Eva branch 1 top-1 acc: {}'.format(results[4])
        print 'Eva branch 1 top-5 acc: {}'.format(results[5])

        print 'Eva branch 2 top-1 acc: {}'.format(results[6])
        print 'Eva branch 2 top-5 acc: {}'.format(results[7])

        print 'Eva branch 3 top-1 acc: {}'.format(results[8])
        print 'Eva branch 3 top-5 acc: {}'.format(results[9])

    if (epoch != 0) and (epoch % 8 == 0):
        K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * 0.96)
