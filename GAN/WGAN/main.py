from utils import get_processed_mnist
from wgan_mnist import WGAN
from train import Trainer

(x_train, y_train), (x_test, y_test) = get_processed_mnist()
model = WGAN()
model.wgan.summary()
model.generator.summary()
model.discriminator.summary()

trainer = Trainer(model)

trainer.fit(x_train, num_epochs=100, print_every=10)
