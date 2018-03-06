from __future__ import division
import os
from glob import glob
import platform
import cPickle
import numpy as np
import pandas as pd
import cv2
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

from semantic_inpainting import Context_Encoder
from utils import load_image, crop_random

isFirstTimeTrain = True

n_epochs = 10000
learning_rate_val = 0.002  # 3e-4
weight_decay_rate = 0.00001
momentum = 0.9
batch_size = 128
lambda_recon = 0.9
lambda_adv = 1 - lambda_recon

overlap_size = 7
hiding_size = 64

if platform.system() == 'Windows':
    trainset_path = 'X:\\DeepLearning\\imagenet_trainset.pickle'
    testset_path = 'X:\\DeepLearning\\imagenet_testset.pickle'
    dataset_path = 'X:\\DeepLearning\\ImageNet_100K'
    result_path = 'E:\\Scholar_Project\\Inpainting\\Context_Encoders\\imagenet'
    model_path = 'E:\\Scholar_Project\\Inpainting\\Context_Encoders\\models\\imagenet'
elif platform.system() == 'Linux':
    trainset_path = '/home/richard/Deep_Learning_Projects/Inpainting/Context_Encoders/imagenet_trainset.pickle'
    testset_path = '/home/richard/Deep_Learning_Projects/Inpainting/Context_Encoders/imagenet_testset.pickle'
    dataset_path = '/home/richard/datasets/ImageNet_100K'
    result_path = '/home/richard/Deep_Learning_Projects/Inpainting/Context_Encoders/imagenet'
    model_path = '/home/richard/Deep_Learning_Projects/Inpainting/Context_Encoders/models/imagenet'

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
    """The reconstruction loss."""
    loss_recon_ori = K.square(y_true - y_pred)
    # loss_recon_center = K.mean(K.sqrt(1e-5 + K.sum(loss_recon_ori * mask_recon, axis=[1, 2, 3]))) / 10.
    # loss_recon_overlap = K.mean(K.sqrt(1e-5 + K.sum(loss_recon_ori * mask_overlap, axis=[1, 2, 3])))
    loss_recon_center = K.sqrt(K.mean(1e-5 + K.sum(loss_recon_ori * mask_recon, axis=[1, 2, 3]))) / 10.
    loss_recon_overlap = K.sqrt(K.mean(1e-5 + K.sum(loss_recon_ori * mask_overlap, axis=[1, 2, 3])))

    return loss_recon_center + loss_recon_overlap


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


context_encoder = Context_Encoder(dis_input_shape=(3, hiding_size, hiding_size),
                                  weight_decay_rate=weight_decay_rate)
encoder_decoder = context_encoder.encoder_decoder
discriminator = context_encoder.discriminator

# reconstruction = context_encoder.recon
reconstruction = encoder_decoder(context_encoder.encoder_input)
dis_output = discriminator(reconstruction)


discriminator.trainable = False
model = Model(inputs=context_encoder.encoder_input,
              outputs=[reconstruction, dis_output],
              name='context_encoder')
print model.name
print model.summary()

# load the model weights, start epoch, iters
if not isFirstTimeTrain:
    model.load_weights(os.path.join(model_path, 'model.h5'))
    encoder_decoder.load_weights(os.path.join(model_path, 'encoder_decoder.h5'))
    discriminator.load_weights(os.path.join(model_path, 'discriminator.h5'))

    with open(os.path.join(model_path, 'start_epoch')) as f:
        start_epoch = cPickle.load(f)

    with open(os.path.join(model_path, 'iters')) as f:
        iters = cPickle.load(f)

else:
    iters = 0  # A batch is an iteration
    start_epoch = 0

    with open(os.path.join(model_path, 'start_epoch'), 'wb') as f:
        cPickle.dump(start_epoch, f, True)

    with open(os.path.join(model_path, 'iters'), 'wb') as f:
        cPickle.dump(iters, f, True)

model_optim = Adam(lr=learning_rate_val, decay=1e-5, clipvalue=10.)
dis_optim = Adam(lr=learning_rate_val * 0.1, decay=1e-5, clipvalue=10.)

model_optim.iterations = K.variable(iters)
dis_optim.iterations = K.variable(iters)

model.compile(loss=[recon_loss, 'binary_crossentropy'],
              loss_weights=[lambda_recon, lambda_adv],
              optimizer=model_optim)


discriminator.trainable = True
discriminator.compile(loss=['binary_crossentropy'],
                      loss_weights=[lambda_adv],
                      optimizer=dis_optim)


testset.index = range(len(testset))
testset = testset.ix[np.random.permutation(len(testset))]


for epoch in range(start_epoch, n_epochs):
    print 'Epoch: {}'.format(epoch + 1)
    trainset.index = range(len(trainset))
    trainset = trainset.ix[np.random.permutation(len(trainset))]

    # Batch
    for start, end in zip(
            range(0, len(trainset), batch_size),
            range(batch_size, len(trainset), batch_size)):

        index = int(start / batch_size)
        image_paths = trainset[start:end]['image_path'].values
        images_ori = map(load_image, image_paths)
        is_none = np.sum([x is None for x in images_ori])
        if is_none > 0:
            continue

        images_crops = map(crop_random, images_ori)
        images, crops, _, _ = zip(*images_crops)

        images = np.array(images)  # images with holes
        crops = np.array(crops)  # the holes cropped from orignal images

        recon_images = encoder_decoder.predict(images)

        X = np.concatenate((crops, recon_images))
        dis_y = [1] * batch_size + [0] * batch_size
        loss_D = discriminator.train_on_batch(X, dis_y)

        # print 'batch {0} d_loss {1}'.format(index, loss_D)

        discriminator.trainable = False
        gen_y = np.ones(batch_size).astype(int)
        loss_G, loss_recon, loss_adv_G = model.train_on_batch([images], [crops, gen_y])
        discriminator.trainable = True

        # print 'batch {0} g_loss {1}, {2}'.format(index, loss_recon, loss_adv_G)

        # Print results every 10 iterations
        if iters % 10 == 0:
            print 'Iter: {}'.format(iters)
            print '...Recon loss: {}'.format(loss_recon)
            print '...Gen adv loss: {}'.format(loss_adv_G)
            print '...Gen loss: {}'.format(loss_G)
            print '...Dis loss: {}'.format(loss_D)

        if iters % 100 == 0:
            test_image_paths = testset[:batch_size]['image_path'].values
            test_images_ori = map(load_image, test_image_paths)

            test_images_crop = [crop_random(image_ori, x=32, y=32) for image_ori in test_images_ori]
            test_images, test_crops, xs, ys = zip(*test_images_crop)

            test_recon_images = encoder_decoder.predict(np.array(test_images))
            test_recon_images = [test_recon_images[i] for i in range(test_recon_images.shape[0])]

            print '============================================================='
            print loss_G, loss_D
            print '============================================================='

            # save the iters evary 100 iterations
            with open(os.path.join(model_path, 'iters'), 'wb') as f:
                cPickle.dump(iters, f, True)

            if iters % 500 == 0:
                ii = 0
                for recon, img, x, y in zip(test_recon_images, test_images, xs, ys):
                    recon_hid = (255. * (recon + 1) / 2.).astype(int)
                    test_with_crop = (255. * (img + 1) / 2.).astype(int)
                    test_with_crop[:, y:y + 64, x:x + 64] = recon_hid
                    test_with_crop = np.transpose(test_with_crop, [1, 2, 0])
                    cv2.imwrite(os.path.join(result_path, 'img_' + str(ii) + '_' +
                                             str(int(iters / 100)) + '.jpg'), test_with_crop)

                    ii += 1
                    if ii > 50:
                        break

                if iters == 0:
                    ii = 0
                    for test_image in test_images_ori:
                        test_image = (255. * (test_image + 1) / 2.).astype(int)
                        test_image[:, 32:32 + 64, 32:32 + 64] = 0
                        test_image = np.transpose(test_image, [1, 2, 0])
                        cv2.imwrite(os.path.join(result_path, 'img_' + str(ii) + '_ori.jpg'), test_image)
                        ii += 1
                        if ii > 50:
                            break

        iters += 1

    discriminator.save_weights(os.path.join(model_path, 'discriminator.h5'))
    encoder_decoder.save_weights(os.path.join(model_path, 'encoder_decoder.h5'))
    model.save_weights(os.path.join(model_path, 'model.h5'))

    # save the epoch
    with open(os.path.join(model_path, 'start_epoch'), 'wb') as f:
        cPickle.dump(epoch + 1, f, True)
