from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2

from keras.optimizers import SGD
from keras.callbacks import Callback
from keras import backend as K

from imagenet_utils import read_batch
from imagenet_utils import crop_random
from imagenet_utils import train_path
from imagenet_utils import val_path

from content_network_vgg import content_network


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

        print('\nThe learning rate is: {:.6f}\n'.format(K.eval(self.model.optimizer.lr)))


n_epochs = 100
n_train_samples = 1281166
n_val_samples = 50000
train_batch_size = 32
init_lr = 0.001

n_train_batches = int(n_train_samples / train_batch_size)

lambda_recon = 0.9
lambda_adv = 1 - lambda_recon

overlap_size = 7
hiding_size = 64

result_path = '/home/richard/Deep_Learning_Projects/Inpainting/my_work/imagenet'

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
opt = SGD(lr=init_lr, momentum=0.9, decay=1e-5)
model.compile(loss=recon_loss,
              optimizer=opt)

for epoch in range(n_epochs):
    print('Epoch: {0}/{1}'.format(epoch + 1, n_epochs))

    for iters in range(n_train_batches):
        print('Iter: {0}/{1}'.format(iters, n_train_batches))

        images, _ = read_batch(batch_size=train_batch_size,
                               images_source=train_path)

        images /= 255.
        images = 2 * images - 1

        images_crops = map(crop_random, images)

        images, crops, xs, ys = zip(*images_crops)

        images = np.array(images)  # images with holes
        crops = np.array(crops)  # the holes cropped from orignal images

        model.fit(x=images,
                  y=crops,
                  batch_size=train_batch_size,
                  epochs=1)

        if (iters != 0)and(iters % 500 == 0):
            print('save the images.')
            ii = 0
            train_recon_images = model.predict(images)
            train_recon_images = [train_recon_images[i] for i in range(train_recon_images.shape[0])]

            for recon, img, x, y in zip(train_recon_images, images, xs, ys):
                recon_hid = (255. * (recon + 1) / 2.).astype(int)
                train_with_crop = (255. * (img + 1) / 2.).astype(int)
                train_with_crop[:, y:y + 64, x:x + 64] = recon_hid
                train_with_crop = np.transpose(train_with_crop, [1, 2, 0])

                cv2.imwrite(os.path.join(result_path, 'img_' + str(ii) + '.jpg'), train_with_crop)
                # http://blog.csdn.net/code_better/article/details/53242943

                ii += 1
                if ii > 50:
                    break
