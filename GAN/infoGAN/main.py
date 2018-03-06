from utils import get_processed_mnist
from infoGAN_mnist import InfoGAN
from trainer import Trainer

(x_train, y_train), (x_test, y_test) = get_processed_mnist()
model = InfoGAN()
model.infogan.summary()
model.discriminator.summary()

trainer = Trainer(model)
trainer.fit(x_train, num_epochs=100, print_every=10)
