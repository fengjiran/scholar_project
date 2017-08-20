import os
import shutil
from glob import glob
import skimage.io
import skimage.transform
import skimage.measure
from PIL import Image

import numpy as np

from keras import backend as K


def load_image(path, pre_height=146, pre_width=146, height=128, width=128):
    try:
        # print path
        img = skimage.io.imread(path).astype(float)
        if img.shape[0] < pre_height or img.shape[1] < pre_width:
            pass
            # print path
            # print img.shape
    except TypeError:
        return None

    img /= 255.

    if img is None:
        return None

    # The shape of image: (height, width, channel)
    if len(img.shape) < 2:
        return None

    if len(img.shape) == 4:
        return None

    if len(img.shape) == 2:
        img = np.tile(img[:, :, None], 3)

    if img.shape[2] == 4:
        img = img[:, :, 3]

    if img.shape[2] > 4:
        return None

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, [pre_height, pre_width])

    rand_y = np.random.randint(0, pre_height - height)
    rand_x = np.random.randint(0, pre_width - width)

    resized_img = resized_img[rand_y:rand_y + height, rand_x:rand_x + width, :]
    resized_img = np.transpose(resized_img, [2, 0, 1])  # convert to channel first

    return resized_img * 2 - 1  # [-1, 1]


def crop_random(image_ori, width=64, height=64, x=None, y=None, overlap=7):

    if image_ori is None:
        return None

    random_y = np.random.randint(overlap, height - overlap) if x is None else x
    random_x = np.random.randint(overlap, width - overlap) if y is None else y

    image = image_ori.copy()
    crop = image_ori.copy()

    crop = crop[:, random_y:random_y + height, random_x:random_x + width]
    image[0,
          random_y + overlap:random_y + height - overlap,
          random_x + overlap:random_x + width - overlap] = 2 * 117. / 255. - 1.

    image[1,
          random_y + overlap:random_y + height - overlap,
          random_x + overlap:random_x + width - overlap] = 2 * 104. / 255. - 1.

    image[2,
          random_y + overlap:random_y + height - overlap,
          random_x + overlap:random_x + width - overlap] = 2 * 123. / 255. - 1.

    return image, crop, random_x, random_y


def search_copy(imageids_path='E:\\Scholar_Project\\Inpainting\\Context_Encoders\\100K_Imagenet_imageIds.txt',
                src_path='X:\\DeepLearning\\ILSVRC2012\\ILSVRC2012_img_train',
                dst_path='X:\\DeepLearning\\ImageNet_100K'):
    """Search the images and copy from src to dst."""
    count_exist = 0
    count_lost = 0
    for filename in open(imageids_path):
        filename = filename.strip('\n')
        foldername = filename.split('_')[0]
        folderpath = os.path.join(src_path, foldername)
        filepath = os.path.join(folderpath, filename)

        if filename in os.listdir(folderpath):
            shutil.copy(filepath, dst_path)
            count_exist += 1
        else:
            with open('lost.txt', 'a') as f:
                f.write(filepath)
                f.write('\n')
            count_lost += 1
        # print filepath

    print count_exist, count_lost


def isValidImage(path='X:\\DeepLearning\\ImageNet_100K'):
    count = 0
    imagenet_images = []
    for filepath, _, _ in os.walk(path):
        imagenet_images.extend(glob(os.path.join(filepath, '*.JPEG')))

    for filename in imagenet_images:
        try:
            Image.open(filename).verify()
        except TypeError:
            print filename
            count += 1
            os.remove(filename)
    print count


# def test():
    # print 'hello world'


# test()

def mse(img1, img2):
    """Compute the mean-squared error between two images.

    Parameters
    ----------
    img1, img2 : ndarray
        Image.  Any dimensionality.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.
    """
    return skimage.measure.compare_mse(img1, img2)


def psnr(img_true, img_test):
    """Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    img_true : ndarray
        Ground-truth image.
    img_test : ndarray
        Test image.

    Returns
    -------
    psnr : float
        The PSNR metric.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    return skimage.measure.compare_psnr(img_true, img_test)


def ssim(img1, img2, multichannel=True):
    return skimage.measure.compare_ssim(img1, img2, multichannel=multichannel)


def ssim_loss(y_true, y_pred):
    """The SSIM loss."""
    k1 = 0.01
    k2 = 0.03
    kernel_size = 11

    data_range = y_true.max() - y_true.min()

    c1 = (k1 * data_range)**2
    c2 = (k2 * data_range)**2

    # if __name__ == '__main__':
    # search_copy()
    # isValidImage()
