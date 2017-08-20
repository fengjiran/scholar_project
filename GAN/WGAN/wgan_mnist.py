from keras.layers import Input, Dense, Reshape, LeakyReLU
from keras.layers import Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K

from utils import sample_unit_gaussian
from utils import wasserstein


class WGAN(object):
    """Class to handle building and training WGAN."""

    def __init__(self,
                 input_shape=(1, 28, 28),
                 latent_dim=100,
                 batch_size=128):
        """
        Setting up everything.

        Parameters
        ----------
        input_shape : Array-like, shape (num_channels, num_rows, num_cols)
            Shape of image.
        latent_dim : int
            Dimension of latent distribution z.
        """
        self.generator = None
        self.discriminator = None
        self.wgan = None

        self.opt_generator = None
        self.opt_discriminator = None

        self.z_input = None
        self.g_output = None
        self.d_input = None
        self.d_output = None

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        self.setup_model()

    def setup_model(self):
        """Method to set up model."""
        self.setup_generator()
        self.setup_discriminator()
        self.setup_wgan()

    def setup_generator(self):
        """Set up generator G."""
        self.z_input = Input(shape=(self.latent_dim,), name='z_input')

        x = Dense(units=256 * 7 * 7)(self.z_input)
        x = Activation('relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Reshape((256, 7, 7))(x)
        x = Dropout(0.5)(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(filters=128,
                            kernel_size=(5, 5),
                            padding='same',
                            data_format=K.image_data_format())(x)

        x = Activation('relu')(x)
        x = BatchNormalization(axis=1, momentum=0.9)(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(filters=64,
                            kernel_size=(5, 5),
                            padding='same',
                            data_format=K.image_data_format())(x)

        x = Activation('relu')(x)
        x = BatchNormalization(axis=1, momentum=0.9)(x)

        x = Conv2DTranspose(filters=32,
                            kernel_size=(5, 5),
                            padding='same',
                            data_format=K.image_data_format())(x)

        x = Activation('relu')(x)
        x = BatchNormalization(axis=1, momentum=0.9)(x)

        x = Conv2DTranspose(filters=1,
                            kernel_size=(5, 5),
                            padding='same',
                            data_format=K.image_data_format())(x)
        self.g_output = Activation('tanh')(x)

        self.generator = Model(inputs=self.z_input,
                               outputs=self.g_output,
                               name='gen_model')

        self.opt_generator = RMSprop(lr=5e-5)
        self.generator.compile(loss=wasserstein,
                               optimizer=self.opt_generator)

    def setup_discriminator(self):
        """Set up discriminator D."""
        self.d_input = Input(shape=self.input_shape, name='d_input')
        x = Conv2D(filters=64,
                   kernel_size=(5, 5),
                   strides=(2, 2),
                   padding='same',
                   activation=LeakyReLU(0.2))(self.d_input)
        x = Dropout(0.5)(x)

        x = Conv2D(filters=64 * 2,
                   kernel_size=(5, 5),
                   strides=(2, 2),
                   padding='same',
                   activation=LeakyReLU(0.2))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(filters=64 * 4,
                   kernel_size=(5, 5),
                   strides=(2, 2),
                   padding='same',
                   activation=LeakyReLU(0.2))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(filters=64 * 8,
                   kernel_size=(5, 5),
                   strides=(2, 2),
                   padding='same',
                   activation=LeakyReLU(0.2))(x)
        x = Dropout(0.5)(x)

        x = GlobalAveragePooling2D()(x)

        self.d_output = Dense(1, name='d_output')(x)

        self.discriminator = Model(inputs=self.d_input,
                                   outputs=self.d_output,
                                   name='dis_model')

        self.opt_discriminator = RMSprop(lr=5e-5)
        self.discriminator.compile(loss=wasserstein,
                                   optimizer=self.opt_discriminator)

    def setup_wgan(self):
        """Setup wgan.

        Discriminator weights should not be trained with the GAN.
        """
        self.discriminator.trainable = False
        wgan_output = self.discriminator(self.g_output)

        self.wgan = Model(inputs=self.z_input,
                          outputs=wgan_output,
                          name='WGAN')

        self.wgan.compile(loss=wasserstein,
                          optimizer=self.opt_generator)

    def sample_latent_distribution(self):
        """Return noise samples from latent distrbution."""
        return sample_unit_gaussian(self.batch_size, self.latent_dim)

    def generate(self):
        """Generate a batch of samples."""
        z = self.sample_latent_distribution()
        return self.generator.predict(z, batch_size=self.batch_size)
