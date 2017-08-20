import glob
import time
import os

import numpy as np

import hickle as hkl

from proc_load import crop_and_mirror


def proc_config(config):
    if not os.path.exists(config['weights_dir']):
        os.makedirs(config['weights_dir'])

        print 'Create folder: ' + config['weights_dir']

    return config


def unpack_config(config, ext_data='.hkl', ext_label='.npy'):

    flag_para_load = config['para_load']

    # Load Training/Validation Filenames and labels
    train_folder = config['train_folder']
    val_folder = config['val_folder']
    label_folder = config['label_folder']

    train_filenames = sorted(glob.glob(train_folder + '/*' + ext_data))
    val_filenames = sorted(glob.glob(val_folder + '/*' + ext_data))
    train_labels = np.load(label_folder + 'train_labels' + ext_label)
    val_labels = np.load(label_folder + 'val_labels' + ext_label)

    img_mean = np.load(config['mean_file'])
    img_mean = img_mean[np.newaxis, :, :, :].astype('float32')

    return (flag_para_load, train_filenames, val_filenames, train_labels, val_labels, img_mean)


def adjust_learning_rate(config, epoch, step_idx, val_record, learning_rate):
    # Adapt learning rate
