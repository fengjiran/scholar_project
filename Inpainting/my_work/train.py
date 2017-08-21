from __future__ import division
from __future__ import print_function

import numpy as np
from keras.optimizers import SGD
from keras import backend as K

from imagenet_utils import read_batch
from imagenet_utils import train_path
from imagenet_utils import val_path

from content_network_vgg import content_network

n_epochs = 100
n_train_samples = 1281166
n_val_samples = 50000
train_batch_size = 128
init_lr = 0.001

n_train_batches = int(n_train_samples / train_batch_size)

lambda_recon = 0.9
lambda_adv = 1 - lambda_recon

overlap_size = 7
hiding_size = 64

# region mask
mask_recon = np.ones((hiding_size - 2 * overlap_size, hiding_size - 2 * overlap_size))
mask_recon = np.lib.pad(mask_recon,
                        pad_width=(overlap_size, overlap_size),
                        mode='constant',
                        constant_values=0)
mask_recon = np.reshape(mask_recon, (1, hiding_size, hiding_size))
mask_recon = np.concatenate([mask_recon] * 3, axis=0)
mask_overlap = 1 - mask_recon


def recon_loss(y_true, y_pred):
    """The reconstuction loss."""
    loss_recon_ori = K.square(y_true - y_pred)
    loss_recon_center = K.mean(K.sqrt(1e-5 + K.sum(loss_recon_ori * mask_recon, axis=[1, 2, 3]))) / 10.
    loss_recon_overlap = K.mean(K.sqrt(1e-5 + K.sum(loss_recon_ori * mask_overlap, axis=[1, 2, 3])))
    # loss_recon_center = K.sqrt(K.mean(1e-5 + K.sum(loss_recon_ori * mask_recon, axis=[1, 2, 3]))) / 10.
    # loss_recon_overlap = K.sqrt(K.mean(1e-5 + K.sum(loss_recon_ori * mask_overlap, axis=[1, 2, 3])))

    return loss_recon_center + loss_recon_overlap


model = content_network()
opt = SGD(lr=init_lr, momentum=0.9)
model.compile(loss=recon_loss,
              optimizer=opt)


for epoch in range(n_epochs):
    print('Epoch: {0}/{1}'.format(epoch + 1, n_epochs))

    for iters in range(n_train_batches):
        images, _ = read_batch(batch_size=train_batch_size,
                               images_source=train_path)
