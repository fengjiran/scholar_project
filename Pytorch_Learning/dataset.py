import platform
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

if platform.system() == 'Windows':
    compress_path = 'E:\\TensorFlow_Learning\\Inpainting\\GlobalLocalImageCompletion_TF\\CelebA\\celeba_train_path_win.pickle'
elif platform.system() == 'Linux':
    compress_path = '/home/richard/TensorFlow_Learning/Inpainting/GlobalLocalImageCompletion_TF/CelebA/celeba_train_path_linux.pickle'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


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
