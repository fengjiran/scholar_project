import os
import json
import random
import platform
from xml.etree import ElementTree
import skimage.io
import skimage.transform
import warnings

# from keras.utils.data_utils import get_file
from keras import backend as K
from keras.utils import to_categorical

import numpy as np

if platform.system() == 'Windows':
    train_path = 'X:\\DeepLearning\\ILSVRC2015\\ILSVRC2015_CLS-LOC\\ILSVRC2015_CLS-LOC\\ILSVRC2015\\Data\\CLS-LOC\\train'
    val_path = 'X:\\DeepLearning\\ILSVRC2015\\ILSVRC2015_CLS-LOC\\ILSVRC2015_CLS-LOC\\ILSVRC2015\\Data\\CLS-LOC\\val'
    test_path = 'X:\\DeepLearning\\ILSVRC2015\\ILSVRC2015_CLS-LOC\\ILSVRC2015_CLS-LOC\\ILSVRC2015\\Data\\CLS-LOC\\test'
    val_annotation_path = 'X:\\DeepLearning\\ILSVRC2015\\ILSVRC2015_CLS-LOC\\ILSVRC2015_CLS-LOC\\ILSVRC2015\\Annotations\\CLS-LOC\\val'
elif platform.system() == 'Linux':
    train_path = '/mnt/DataBase/DeepLearning/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/train'
    val_path = '/mnt/DataBase/DeepLearning/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/val'
    val_annotation_path = '/mnt/DataBase/DeepLearning/ILSVRC2015/ILSVRC2015_CLS-LOC/ILSVRC2015_CLS-LOC/ILSVRC2015/Annotations/CLS-LOC/val'


# CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


def read_class_name_discription(path='imagenet_class_index.json'):
    CLASS_INDEX = json.load(open(path))
    class_name = []  # i.e. tench
    class_discription = []  # i.e. n01440764

    for i in range(len(CLASS_INDEX)):
        class_discription.append(str(CLASS_INDEX[str(i)][0]))
        class_name.append(str(CLASS_INDEX[str(i)][1]))

    return class_name, class_discription


class_name, class_discription = read_class_name_discription()


def load_single_image(path, pre_height=256, pre_width=256, height=224, width=224):
    img = skimage.io.imread(path).astype(float)
    if img.shape[0] < pre_height or img.shape[1] < pre_width:
        return None

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

    return resized_img  # [3, 224, 224]


def read_batch(batch_size, images_source):
    """It returns a batch of single images (no data-augmentation).

    ILSVRC 2012 training set folder should be srtuctured like this:
    ILSVRC2012_img_train
        |_n01440764
        |_n01443537
        |_n01484850
        |_n01491361
        |_ ...
    Args:
        batch_size: need explanation? :)
        images_sources: path to ILSVRC 2012 training set folder
        wnid_labels: list of ImageNet wnid lexicographically ordered
    Returns:
        batch_images: a tensor (numpy array of images) of shape [batch_size, channels, width, height]
        batch_labels: a tensor (numpy array of vectors) of shape [batch_size, 1000]
    """
    batch_images = []
    batch_labels = []

    # class_name, class_discription = read_class_name_discription()

    ind = 0
    while ind < batch_size:
        # random class choise
        # randomly choose a folder of image of the same class from a list of previously sorted wnids
        class_ind = random.randint(0, 999)
        folder = class_discription[class_ind]  # a string
        folder_path = os.path.join(images_source, folder)
        img_path = os.path.join(folder_path, random.choice(os.listdir(folder_path)))
        img = load_single_image(img_path)
        if img is not None:
            batch_images.append(img)
            batch_labels.append(class_ind)
            ind += 1

    # batch_images = np.array(batch_images)
    # batch_labels = np.array(batch_labels)

    return np.array(batch_images), np.array(batch_labels)


def read_validation_batch(batch_size, val_path, val_annotation_path):
    batch_images_val = []
    batch_labels_val = []

    # class_name, class_discription = read_class_name_discription()

    images_val = sorted(os.listdir(val_path))

    i = 0
    while i < batch_size:
        idx = random.randint(0, len(images_val) - 1)
        image_name = images_val[idx]
        img_path = os.path.join(val_path, image_name)
        img = load_single_image(img_path)
        if img is not None:
            batch_images_val.append(img)

            # parse the annotation xml file
            annotation_name = image_name.split('.')[0] + '.xml'
            f = ElementTree.parse(os.path.join(val_annotation_path, annotation_name))
            root = f.getroot()
            obj = root.find('object')
            dis = obj.find('name').text
            class_ind = class_discription.index(dis)
            batch_labels_val.append(class_ind)
            i += 1

    return np.array(batch_images_val), np.array(batch_labels_val)


def preprocess_input(x, data_format=None):
    """Preprocesse a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68

    return x


def imagenet_size(train_path):
    """It calculates the number of examples in ImageNet training-set.

    Args:
        path: path to ILSVRC training set folder
    Returns:
        n: the number of training examples
    """
    n = 0
    for d in os.listdir(train_path):
        for f in os.listdir(os.path.join(train_path, d)):
            n += 1

    return n


def decode_predictions(preds, top=5):
    """Decode the prediction of an ImageNet model.

    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: integer, how many top-guesses to return.

    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.

    # Raises
        ValueError: in case of invalid shape of the `pred` array
            (must be 2D).
    """
    # global CLASS_INDEX
    CLASS_INDEX = json.load(open('imagenet_class_index.json'))
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    # if CLASS_INDEX is None:
        # fpath = get_file('imagenet_class_index.json',
        #                  CLASS_INDEX_PATH,
        #                  cache_subdir='models')
        # CLASS_INDEX = json.load(open(fpath))

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        include_top,
                        weights='imagenet'):
    """Internal utility to compute/validate an ImageNet model's input shape.

    # Arguments
        input_shape: either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: default input width/height for the model.
        min_size: minimum input width/height accepted by the model.
        data_format: image data format to use.
        include_top: whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: one of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.

    # Returns
        An integer shape tuple (may include None entries).

    # Raises
        ValueError: in case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape is not None and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] != 3 and input_shape[0] != 1:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed ' + str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] != 3 and input_shape[-1] != 1:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed ' + str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if include_top:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True`, '
                                 '`input_shape` should be ' + str(default_shape) + '.')
        input_shape = default_shape
    else:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError('`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                        (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) + ', got '
                                     '`input_shape=' + str(input_shape) + '`')
            else:
                input_shape = (3, None, None)
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError('`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                        (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) + ', got '
                                     '`input_shape=' + str(input_shape) + '`')
            else:
                input_shape = (None, None, 3)
    return input_shape


if __name__ == '__main__':
    # name, dis = read_class_name_discription()
    a, b = read_batch(128, train_path)
    # a, b = read_validation_batch(256, val_path, val_annotation_path)
    batch = preprocess_input(a)
    b = to_categorical(b, 1000)
    print batch.shape
    print np.max(batch), np.min(batch)
    # print imagenet_size(train_path)
    print b.shape
    # print b
