from keras.layers import Input, Dense, Reshape, LeakyReLU
from keras.layers import concatenate, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras import backend as K


from utils import sample_unit_gaussian
from utils import sample_categorical
from utils import disc_mutual_info_loss


class InfoGAN(object):
    """Class to handle building and training infoGAN models."""

    def __init__(self,
                 input_shape=(1, 28, 28),
                 latent_dim=62,
                 reg_cont_dim=0,
                 reg_disc_dim=10,
                 batch_size=128):
        """
        Setting up everything.

        Parameters
        ----------
        input_shape : Array-like, shape (num_channels, num_rows, num_cols)
            Shape of image.
        latent_dim : int
            Dimension of latent distribution z.
        reg_cont_dim : int
            Dimension of continuous latent regularized distribution c.
        reg_disc_dim : int
            Dimension of discrete latent regularized distribution c.
        """
        self.generator = None
        self.discriminator = None
        self.auxiliary = None
        self.infogan = None

        self.opt_generator = None
        self.opt_discriminator = None

        self.z_input = None
        self.c_disc_input = None
        self.g_output = None
        self.d_input = None

        self.d_hidden = None  # Store this to set up Q
        self.d_output = None
        self.q_output = None

        self.input_shape = input_shape
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.reg_cont_dim = reg_cont_dim
        self.reg_disc_dim = reg_disc_dim

        self.total_latent_dim = latent_dim + reg_cont_dim + reg_disc_dim

        self.setup_model()

    def setup_model(self):
        """Method to set up model."""
        self.setup_generator()
        self.setup_discriminator()
        self.setup_auxiliary()
        self.setup_infogan()

    def setup_generator(self):
        """Set up generator G."""
        self.z_input = Input(shape=(self.latent_dim,), name='z_input')
        self.c_disc_input = Input(shape=(self.reg_disc_dim,), name='c_input')

        x = concatenate([self.z_input, self.c_disc_input], axis=1)
        x = Dense(units=256 * 7 * 7)(x)
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

        self.generator = Model(inputs=[self.z_input, self.c_disc_input],
                               outputs=self.g_output,
                               name='gen_model')

        self.opt_generator = RMSprop(lr=1e-4, clipvalue=1.0, decay=6e-8)
        # self.opt_generator = Adam(lr=1e-3)
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=self.opt_generator)
        # print self.generator.summary()

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
        self.d_hidden = Flatten()(x)
        self.d_output = Dense(1, activation='sigmoid', name='d_output')(self.d_hidden)

        self.discriminator = Model(inputs=self.d_input,
                                   outputs=self.d_output,
                                   name='dis_model')

        self.opt_discriminator = RMSprop(lr=5e-4, clipvalue=1.0, decay=3e-8)
        # self.opt_discriminator = Adam(lr=2e-4)
        # self.opt_discriminator = SGD(lr=1e-4, momentum=0.9, nesterov=True)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.opt_discriminator)
        # print self.discriminator.summary()

    def setup_auxiliary(self):
        """Setup auxiliary distribution."""
        x = Dense(128)(self.d_hidden)
        x = LeakyReLU(0.1)(x)
        self.q_output = Dense(units=self.reg_disc_dim,
                              activation='softmax',
                              name='auxiliary')(x)
        self.auxiliary = Model(inputs=self.d_input,
                               outputs=self.q_output,
                               name='aux_model')

        # It does not matter what the loss is here, as we do not specifically train this model
        self.auxiliary.compile(loss='mse', optimizer=self.opt_discriminator)

        # print self.auxiliary.summary()

    def setup_infogan(self):
        """Setup infoGAN.

        Discriminator weights should not be trained with the GAN.
        """
        self.discriminator.trainable = False
        gan_output = self.discriminator(self.g_output)
        gan_output_aux = self.auxiliary(self.g_output)

        self.infogan = Model(inputs=[self.z_input, self.c_disc_input],
                             outputs=[gan_output, gan_output_aux])

        self.infogan.compile(loss={'dis_model': 'binary_crossentropy',
                                   'aux_model': disc_mutual_info_loss},
                             loss_weights={'dis_model': 1.,
                                           'aux_model': -1.},
                             optimizer=self.opt_generator)

    def sample_latent_distribution(self):
        """Return continuous and discrete samples from latent distrbution."""
        z = sample_unit_gaussian(self.batch_size, self.latent_dim)
        c_disc = sample_categorical(self.batch_size, self.reg_disc_dim)
        return z, c_disc

    def generate(self):
        """Generate a batch of samples."""
        z, c_disc = self.sample_latent_distribution()
        return self.generator.predict([z, c_disc], batch_size=self.batch_size)

    def discriminate(self, x_batch):
        return self.discriminator.predict(x_batch, batch_size=self.batch_size)

    def get_aux_dist(self, x_batch):
        return self.auxiliary.predict(x_batch, batch_size=self.batch_size)
