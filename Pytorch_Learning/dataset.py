import platform
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms


if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\celeba_train_path_win.pickle'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/celeba_train_path_linux.pickle'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def input_transfrom(image_path):
    low = 48
    high = 96
    image_height = 178
    image_width = 178
    gt_height = 96
    gt_width = 96

    img = Image.open(image_path).convert('RGB')
    img.show()
    ori_image = transforms.RandomCrop((image_height, image_width))(img)  # range [0, 255]

    hole_height, hole_width = np.random.randint(low, high, size=(2))
    y = np.random.randint(0, image_height - hole_height)
    x = np.random.randint(0, image_width - hole_width)

    mask = np.ones((hole_height, hole_width), dtype=np.float32)
    mask = np.pad(mask,
                  ((y, image_height - hole_height - y), (x, image_width - hole_width - x)),
                  'constant')
    mask = np.reshape(mask, [image_height, image_width, 1])
    mask = np.concatenate([mask] * 3, 2)

    x_loc = np.random.randint(low=np.max([0, x + hole_width - gt_width]),
                              high=np.min([x, image_width - gt_width]) + 1)
    y_loc = np.random.randint(low=np.max([0, y + hole_height - gt_height]),
                              high=np.min([y, image_height - gt_height]) + 1)

    image_with_hole = np.array(ori_image) * (1 - mask) + mask * 255.  # range [0,255]

    tsf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    ori_image = tsf(ori_image)  # range [-1,1]
    image_with_hole = tsf(image_with_hole)  # range [-1, 1]
    mask = transforms.ToTensor()(mask * 255)

    return ori_image, image_with_hole, mask, x_loc, y_loc


class CelebaDataset(Dataset):
    """Construc celeba dataset."""

    def __init__(self, image_filename_dir, input_transfrom=None):
        super(CelebaDataset, self).__init__()
        train_path = pd.read_pickle(image_filename_dir)
        train_path.index = range(len(train_path))
        train_path = train_path.ix[np.random.permutation(len(train_path))]
        self.train_path = train_path[:]['image_path'].values.tolist()

        self.input_transfrom = input_transfrom

    def __getitem__(self, index):
        """Get one item of dataset."""
        pass

    def __len__(self):
        """Get the number of samples in dataset."""
        return len(self.train_path)


if __name__ == '__main__':
    img_path = '/Users/apple/Desktop/richard/Tensorflow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/000013.png'
    ori_image, image_with_hole, mask, x_loc, y_loc = input_transfrom(img_path)

    print(ori_image.size())
    print(image_with_hole.size())
    print(mask.size())
    print(x_loc, y_loc)

    # ori_image = transforms.ToPILImage()(ori_image)
    # image_with_hole = transforms.ToPILImage()(image_with_hole)
    # mask = transforms.ToPILImage()(mask)
    # ori_image.show()
    # image_with_hole.show()
    # mask.show()
