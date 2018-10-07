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


def get_all_directories():
    return {
        # 'data': '../../data',
            'pred': 'pred',
            'out': 'output',
            'vis': '/home/ankit/Pictures/unet_position',
            'tlog': 'logs/train',
            'plog': 'logs/pred',
            'weights': 'weights',
            'augmented': 'augmented'}


def create_environment():
    dirs = get_all_directories()
    # if not os.path.isdir(dirs['data']):
    #     print('Error: data directory not found. The data folder must be located one level above the project folder.')
    #     exit(1)
    if not os.path.isdir('configs'):
        os.mkdir('configs')
    if not os.path.isdir(dirs['pred']):
        os.mkdir(dirs['pred'])
    if not os.path.isdir(dirs['out']):
        os.mkdir(dirs['out'])
    if not os.path.isdir(dirs['tlog']):
        os.makedirs(dirs['tlog'])
    if not os.path.isdir(dirs['plog']):
        os.makedirs(dirs['plog'])
    if not os.path.isdir(dirs['weights']):
        os.mkdir(dirs['weights'])
    if not os.path.isdir(dirs['augmented']):
        os.mkdir(dirs['augmented'])


def get_all_images():
    dirs = get_all_directories()

    data = {}
    seqs = sorted(os.listdir(dirs['data']))
    for seq in seqs:
        seq_dir = os.path.join(dirs['data'], seq)

        img_paths = sorted(os.listdir(os.path.join(seq_dir, 'imgs')))
        msk_paths = sorted(os.listdir(os.path.join(seq_dir, 'msks')))
        pts_files = sorted(os.listdir(os.path.join(seq_dir, 'pts')))

        needle_paths = []
        plastic_paths = []
        for pt_file in pts_files:
            if 'needle' in pt_file:
                needle_paths.append(os.path.join(seq_dir, 'pts/' + pt_file))
            if 'plastic' in pt_file:
                plastic_paths.append(os.path.join(seq_dir, 'pts/' + pt_file))

        needle_pts = []
        plastic_pts = []
        for pt_file in sorted(needle_paths):
            with open(pt_file) as f:
                needle_pts.extend(f.readlines())
        for pt_file in sorted(plastic_paths):
            with open(pt_file) as f:
                plastic_pts.extend(f.readlines())

        d = dict()
        for j in range(len(img_paths)):
            img_hash = compute_img_hash(img_paths[j])
            d[img_hash] = {'img': os.path.join(seq_dir, 'imgs/' + img_paths[j]),
                           'msk': os.path.join(seq_dir, 'msks/' + msk_paths[j]),
                           'needle': needle_pts[j],
                           'plastic': plastic_pts[j]}

        data.update(d)

    return data


def find_test_images(config_hash):
    dirs = get_all_directories()
    data = get_all_images()

    test_images = dict()

    for pred_instance in os.listdir(dirs['pred']):
        if int(pred_instance.split('_')[1]) == config_hash:
            instance_dir = os.path.join(dirs['pred'], pred_instance)

            for prd in sorted(os.listdir(instance_dir)):
                token = re.split(regex, prd)
                prd_hash = compute_img_hash(prd)

                if prd_hash in test_images:
                    continue
                else:
                    if 'left' in prd:
                        n_prd_path = 'needle_left_' + token[2] + '_' + token[3] + '.png'
                        p_prd_path = 'plastic_left_' + token[2] + '_' + token[3] + '.png'
                    if 'right' in prd:
                        n_prd_path = 'needle_right_' + token[2] + '_' + token[3] + '.png'
                        p_prd_path = 'plastic_right_' + token[2] + '_' + token[3] + '.png'
                    test_image_data = data[prd_hash]
                    test_image_data['needle_prd_path'] = os.path.join(instance_dir, n_prd_path)
                    test_image_data['plastic_prd_path'] = os.path.join(instance_dir, p_prd_path)
                    test_images[prd_hash] = data[prd_hash]

    return test_images


def compute_img_hash(img_name):
    """
    This function takes the image name as an input and computes a hash code for that image
    :param img_name: The image name in the form "side_seq_XXXX.png"
    :return: the hash code
    """
    token = re.split(regex, img_name)
    hash = 0
    if 'left' in token:
        hash = int(token[-3]) * 10000 + int(token[-2]) * 10 + 0
    if 'right' in token:
        hash = int(token[-3]) * 10000 + int(token[-2]) * 10 + 1
    return hash



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


if __name__ == '__main__':
    print(find_test_images(0))
