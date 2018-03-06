import numpy as np
from PIL import Image
from utils import get_batch
from utils import combine_images


class Trainer(object):
    """Class to train the infoGAN model."""

    def __init__(self, model):
        self.model = model
        self.d_loss_history = []
        self.g_loss_history = []  # infogan loss
        self.ent_loss_history = []
        self.generated_image = None

    def train_gan(self):
        z, c_disc = self.model.sample_latent_distribution()
        # Generate examples that fool classifier, i.e. where D outputs 1
        target = np.ones(self.model.batch_size).astype(int)
        infogan_loss = self.model.infogan.train_on_batch([z, c_disc], [target, c_disc])
        self.g_loss_history.append(infogan_loss[1])
        self.ent_loss_history.append(infogan_loss[2])

    def train_discriminator(self, x_batch=None):
        real_image = x_batch
        fake_image = self.model.generate()
        self.generated_image = fake_image

        X = np.concatenate((real_image, fake_image))
        y = [1] * self.model.batch_size + [0] * self.model.batch_size
        d_loss = self.model.discriminator.train_on_batch(X, y)
        self.d_loss_history.append(d_loss)

        # # Train discriminator on fake data
        # if x_batch is None:
        #     fake_batch = self.model.generate()
        #     # Fake examples, so D should output 0
        #     target = np.zeros(self.model.batch_size).astype(int)
        #     d_loss = self.model.discriminator.train_on_batch(fake_batch, target)
        # # Or train discriminator on real data
        # else:
        #     # Real data, so D should output 1
        #     target = np.ones(self.model.batch_size).astype(int)
        #     d_loss = self.model.discriminator.train_on_batch(x_batch, target)
        # self.d_loss_history.append(d_loss)

    def fit(self, x_train, num_epochs=1, print_every=0):
        """Method to train GAN.

        Parameters
        ----------
        print_every : int
            Print loss information every |print_every| number of batches. If 0
            prints nothing.
        """
        num_batches = x_train.shape[0] / self.model.batch_size
        print 'num batches {}'.format(num_batches)

        for epoch in range(num_epochs):
            print '\nEpoch {}'.format(epoch + 1)

            for index in range(num_batches):
                x_batch = get_batch(x_train, self.model.batch_size)
                self.train_discriminator(x_batch)
                self.train_gan()

                if print_every and (index % print_every) == 0:
                    print 'GAN loss {}'.format(self.g_loss_history[-1])
                    print 'D loss {}'.format(self.d_loss_history[-1])
                    print 'Entropy loss {}'.format(self.ent_loss_history[-1])
                    print '\n'

            image = combine_images(self.generated_image)
            image = image * 127.5 + 127.5

            Image.fromarray(image.astype(np.uint8)).save('./generated_mnist/' + str(epoch) + ".png")

            self.model.generator.save_weights('generator.h5', True)
            self.model.infogan.save_weights('infogan.h5', True)
