import os
import datetime
import numpy as np
import re
import math


delimiters = '.', '_', '/'
regex = '|'.join(map(re.escape, delimiters))


def to_image_range(array):
    """
    This method maps the value range of any given array to the value range (0, 255).
    :param array: the input array
    :return: the linearly mapped output array with the same shape as the input array and value
             range (0, 255).
    """
    old_min = np.min(array)
    old_max = np.max(array)
    old_range = old_max - old_min
    new_range = 255
    arr_norm = [np.maximum(0, np.minimum(255, ((x - old_min) * new_range / old_range))) for x in array]
    return np.reshape(arr_norm, array.shape).astype(np.uint8)


def to_rgb(img):
    """
    This method converts a grayscale image into an rgb image.
    :param img: the grayscale input image
    :return: rgb output image
    """
    out = np.ndarray((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    out[:, :, 0] = img
    out[:, :, 1] = img
    out[:, :, 2] = img
    return out


def to_gray(img):
    """
    This method converts an rgb image into a grayscale image
    :param img: rgb input image
    :return: grayscale output image
    """
    out = np.ndarray((img.shape[0], img.shape[1]), dtype=np.uint8)
    out[:, :] = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3
    return out


def transform_from_axisangle(axis, angle, translation):
    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = unit_vector(axis)
    rot = np.diag([cosa, cosa, cosa])
    rot += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    rot += np.array([[0.0, -axis[2], axis[1]],
                   [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]])
    transform = np.identity(4)
    transform[:3, :3] = rot
    transform[:3, 3] = translation
    return transform


def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def get_date_string():
    """
    This method creates a string containing the current date and time
    :return: A string containing the current date and time
    """
    dt = datetime.datetime.now()
    return str(dt.year) + '-' + str(dt.month) + '-' + str(dt.day) \
        + '_' + str(dt.hour) + ':' + str(dt.minute) + ':' + str(dt.second)


def save_mask_and_pred(self, mask, pred):
    save_folder = 'pred/'
    mask = to_image_range(mask.cpu().detach().numpy())
    pred = to_image_range(pred.cpu().detach().numpy())
    mask_pred_pair = np.hstack((mask, pred))
    result = Image.fromarray(mask_pred_pair)
    result.save(save_folder + 'mask_' + str(self.saved_images) + '.png')
