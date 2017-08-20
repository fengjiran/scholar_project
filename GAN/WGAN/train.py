import numpy as np
from PIL import Image

# from keras.optimizers import RMSprop

from utils import get_batch
from utils import combine_images
# from utils import wasserstein


class Trainer(object):
    """Class to train the WGAN model."""

    def __init__(self, model):
        self.model = model
        self.d_loss_history = []
        self.wgan_loss_history = []

        self.clip_low = -0.01
        self.clip_high = 0.01
        self.dis_iterations = 5

        # self.opt_G = RMSprop(lr=5e-5)
        # self.opt_D = RMSprop(lr=5e-5)

        self.generated_image = None

    def train_discriminator(self, real_image):
        fake_image = self.model.generate()
        # self.generated_image = fake_image

        X = np.concatenate((real_image, fake_image))
        y = [-1] * self.model.batch_size + [1] * self.model.batch_size
        d_loss = self.model.discriminator.train_on_batch(2 * X, y)
        self.d_loss_history.append(d_loss)

    def train_wgan(self):
        z = self.model.sample_latent_distribution()
        y = [-1] * self.model.batch_size

        wgan_loss = self.model.wgan.train_on_batch(z, y)
        self.wgan_loss_history.append(wgan_loss)

    def fit(self, x_train, num_epochs=1, print_every=0):
        """Method to train WGAN."""
        num_batches = x_train.shape[0] / self.model.batch_size
        print 'num batches {}'.format(num_batches)

        gen_iterations = 0

        for epoch in range(num_epochs):
            print '\nEpoch {}'.format(epoch + 1)

            for index in range(num_batches):

                # if gen_iterations < 25 or gen_iterations % 500 == 0:
                #     self.dis_iterations = 100

                # train the critic / discriminator
                for disc_it in range(self.dis_iterations):

                    # clip discriminator weights
                    for l in self.model.discriminator.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, self.clip_low, self.clip_high) for w in weights]
                        l.set_weights(weights)

                    x_batch = get_batch(x_train, self.model.batch_size)

                    self.train_discriminator(x_batch)

                # train the generator
                self.train_wgan()

                gen_iterations += 1

                if print_every and (index % print_every) == 0:
                    print 'WGAN loss: {}'.format(self.wgan_loss_history[-1])
                    print 'D loss: {}'.format(self.d_loss_history[-1])
                    print '\n'

            self.generated_image = self.model.generate()
            image = combine_images(self.generated_image)
            image = image * 127.5 + 127.5
            Image.fromarray(image.astype(np.uint8)).save('./generated_mnist/' + str(epoch) + ".png")

            self.model.generator.save_weights('generator.h5', True)
            self.model.wgan.save_weights('wgan.h5', True)
