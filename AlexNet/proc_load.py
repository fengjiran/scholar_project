'''
Load data in parallel with train.py
'''
import time
import math

import numpy as np
import zmq
import hickle as hkl


def get_params_crop_and_mirror(param_rand, data_shape, cropsize):

    center_margin = (data_shape[3] - cropsize) / 2
    crop_xs = round(param_rand[0] * center_margin * 2)
    crop_ys = round(param_rand[1] * center_margin * 2)

    if False:
        crop_xs = math.floor(param_rand[0] * center_margin * 2)
        crop_ys = math.floor(param_rand[1] * center_margin * 2)

    flag_mirror = bool(round(param_rand[2]))

    return crop_xs, crop_ys, flag_mirror


def crop_and_mirror(data, param_rand, flag_batch=True, cropsize=227):
    '''
    when param_rand == (0.5, 0.5, 0), it means no randomness
    '''

    # if param_rand == (0.5, 0.5, 0), means no randomness and do validation
    if param_rand[0] == 0.5 and param_rand[1] == 0.5 and param_rand[2] == 0:
        flag_batch = True

    if flag_batch:
        # mirror and crop the whole batch
        crop_xs, crop_ys, flag_mirror = get_params_crop_and_mirror(param_rand, data.shape, cropsize)

        # random mirror
        if flag_mirror:
            data = data[:, :, :, ::-1]

        # random crop
        data = data[:, :, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize]

    else:
        # mirror and crop each batch individually
        # to ensure consistency, use the param_rand[1] as seed
        np.random.seed(int(10000 * param_rand[1]))

        data_out = np.zeros((data.shape[0], data.shape[1], cropsize, cropsize)).astype('float32')

        for ind in range(data.shape[0]):
            # generate random numbers
            tmp_rand = np.float32(np.random.rand(3))
            tmp_rand[2] = round(tmp_rand[2])

            # get mirror/crop parameters
            crop_xs, crop_ys, flag_mirror = get_params_crop_and_mirror(tmp_rand, data.shape, cropsize)

            # do image crop and mirror
            img = data[ind, :, :, :]
            if flag_mirror:
                img = img[:, :, ::-1]

            img = img[:, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize]
            data_out[ind, :, :, :] = img

        data = data_out

    return np.ascontiguousarray(data, dtype='float32')
