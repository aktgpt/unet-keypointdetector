import os
import datetime
import numpy as np
import re
import math
import cv2
import copy
from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError

delimiters = '.', '_', '/'
regex = '|'.join(map(re.escape, delimiters))

class PrecisionRecall(Metric):
    """
    Calculates precision.

    - `update` must receive output of the form `(y_pred, y)`.

    If `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the precision for each class.
    """
    def __init__(self, ):
        super(PrecisionRecall, self).__init__()

    def reset(self):
        self.all_positives = []
        self.true_positives = []
        self.false_positives = []
        self.false_negatives = []
        self.euclidean_distance = []
        self.more_than_normal_maximas = 0
        self.no_maxima_images = 0

    def update(self, output):
        pred = output[0].cpu().data.numpy()
        mask = output[1].cpu().data.numpy()
        threshold = 99.2
        for i in range(pred.shape[0]):
            try:
                points_pred= find_local_maxima(pred[i, 0, :, :], threshold)
                points_true = find_local_maxima(mask[i, 0, :, :], threshold)
                if len(points_true)>6:
                    self.more_than_normal_maximas =self.more_than_normal_maximas + 1
                    # print("too many detections. Maximas detected: " + str(len(points_true)))
                for i in range(points_true.shape[0]):
                    deltas = points_pred - points_true[i]
                    dist = np.sqrt(np.sum(deltas ** 2, axis=1))
                    min_ind = np.argmin(dist)
                    if dist[min_ind] > 5:
                        self.false_positives.append(1)
                    else:
                        self.true_positives.append(1)
                        self.euclidean_distance.append(dist[min_ind])
                for i in range(points_pred.shape[0]):
                    deltas = points_true - points_pred[i]
                    dist = np.sqrt(np.sum(deltas ** 2, axis=1))
                    min_ind = np.argmin(dist)
                    if dist[min_ind] > 5:
                        self.false_negatives.append(1)
            except:
                self.no_maxima_images = self.no_maxima_images + 1
                # print("no maximas found")
                pass

    def compute(self):
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        fn = len(self.false_negatives)
        print("no maxima image: " +str(self.no_maxima_images))
        print("more than expected maxima images:" + str(self.more_than_normal_maximas))
        if tp+fn is not 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            mean_euclidean_dist = np.mean(self.euclidean_distance)
            if self.all_positives is None:
                raise NotComputableError('Precision must have at least one example before it can be computed')
            result = {'precision': precision,
                      'recall': recall,
                      'mean_euclidean_dist': mean_euclidean_dist}
        else:
            result = {'precision': None,
                      'recall': None,
                      'mean_euclidean_dist': None}
        return result


def find_local_maxima(image, threshold):
    thresh_image = copy.deepcopy(image)
    percentile_thres = np.percentile(image, threshold)
    thresh_image[image <= percentile_thres] = 0
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    strel3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    strel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    strel7x7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img_dilate3 = cv2.dilate(thresh_image, strel3x3)
    img_dilate5 = cv2.dilate(thresh_image, strel5x5)
    img_dilate7 = cv2.dilate(thresh_image, strel7x7)
    dst3 = cv2.compare(image, img_dilate3, cv2.CMP_EQ)
    dst5 = cv2.compare(image, img_dilate5, cv2.CMP_EQ)
    dst7 = cv2.compare(image, img_dilate7, cv2.CMP_EQ)
    dst_temp = np.logical_and(dst3, dst5)
    dst = np.logical_and(dst_temp, dst7)
    y, x = np.where(dst == True)
    detections = np.zeros((len(x), 2))
    idx = x.argsort()
    detections[:, 0] = x[idx]
    detections[:, 1] = y[idx]
    return detections

def match_predicted_points(points_pred, points_true):
    closest_points = np.empty(points_true.shape)
    for i in range(points_true.shape[0]):
        deltas = points_pred - points_true[i]
        dist = np.sqrt(np.sum(deltas**2, axis=1))
        min_ind = np.argmin(dist)
        if dist[min_ind] > 5:
            closest_points[i] = np.array([np.nan, np.nan])
        else:
            closest_points[i] = points_pred[min_ind]
    return closest_points, points_true


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
