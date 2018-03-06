from keras.optimizers import RMSprop
from keras.utils import to_categorical

from infoGAN_mnist import InfoGAN
from utils import sample_unit_gaussian
from utils import combine_images

import numpy as np
from PIL import Image

batch_size = 16
latent_dim = 62
reg_disc_dim = 10

model = InfoGAN()
generator = model.generator
opt_generator = RMSprop(lr=1e-4, clipvalue=1.0, decay=6e-8)
generator.compile(loss='binary_crossentropy',
                  optimizer=opt_generator)

generator.load_weights('generator.h5')

z = sample_unit_gaussian(batch_size, latent_dim)
c_disc = to_categorical([8] * batch_size, reg_disc_dim).astype('int8')

generated_images = generator.predict([z, c_disc])
image = combine_images(generated_images)

image = image * 127.5 + 127.5
Image.fromarray(image.astype(np.uint8)).save("generated_image.png")
