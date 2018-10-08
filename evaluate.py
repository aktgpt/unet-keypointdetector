import json
import argparse
from unet import UNet
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import cv2
import numpy as np
import glob
import matplotlib.pylab as plt
import copy
import time
from utility import to_image_range

argparser = argparse.ArgumentParser(
    description='Train U-net dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, dropout, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.dropout = dropout

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.dropout(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, dropout,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.dropout = dropout

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.dropout(x)
        return x

class UNet(nn.Module):
    def __init__(self, config):
        self.input_shape = tuple(config['unet']['input_shape'])
        self.num_classes = config['unet']['n_classes']
        self.start_channels = config['unet']['start_ch']
        self.depth = config['unet']['depth']
        up_mode = config['unet']['up_mode']
        merge_mode = config['unet']['merge_mode']

        super(UNet, self).__init__()
        self.dropout = nn.Dropout(config['unet']['dropout'])

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.in_channels = self.input_shape[2]

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(self.depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_channels * (2 ** i)
            pooling = True if i < self.depth - 1 else False

            down_conv = DownConv(ins, outs, self.dropout, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(self.depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, self.dropout, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)
        x = self.dropout(x)
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


class MultipleKeypointDetector:
    def __init__(self, config_path):
        self.config = config_path
        self.threshold = 0
        print("init")
        self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path) as config_buffer:
            config = json.load(config_buffer)
        self.unet_input_shape = tuple(config['unet']['input_shape'])
        self.weight_path = config['weights_path']
        self.model = UNet(config)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def init(self):
        # self.summary()
        self.load_weights(self.weight_path)
        self.summary()

    def load_weights(self, path):
        self.model.cuda().float()
        self.model.load_state_dict(torch.load(path))

    def summary(self):
        print(self.model.summary())

    def predict(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.model.eval()
        self.img_shape = np.shape(image)
        img = cv2.resize(image, (self.unet_input_shape[1], self.unet_input_shape[0]))
        img = img.transpose((2, 0, 1))
        img = img[np.newaxis, ...]
        # img_2 = np.concatenate((img, img, img, img), axis=0)
        start = time.time()
        img = torch.from_numpy(img).float().cuda()
        # img_2 = img_2.type(torch.cuda.HalfTensor)
        pred = self.model(img)
        pred_np = pred.cpu().data.numpy()[0, 0, :, :]
        end = time.time()
        print(end-start)

        pred_np = to_image_range(pred_np)
        points = self.keypoints_location(pred_np)
        # prediction_image = self.to_image_range(pred_image)
        return points

    def convert_predict_to_gray(self, prediction):
        pred_image = prediction[0, :, :, 0]
        mask = cv2.resize(pred_image, (self.img_shape[1], self.img_shape[0]))
        return mask

    def keypoints_location(self, image, threshold):
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


if __name__ == '__main__':
    args = argparser.parse_args(['-c', 'configs/config_keypoints.json'])
    config_path = args.conf
    # img_folder = '/home/ankit/Documents/Ankit-BackUp/Develop/vbtrack/data/calibration_images/left/'
    img_folder = '/home/ankit/Documents/Ankit-BackUp/Develop/vbtrack/data/unet_images/'
    images = glob.glob(img_folder+"*.png")

    keypoint_detector = MultipleKeypointDetector(config_path)
    keypoint_detector.init()

    for i in range(len(images)):
        image = cv2.imread(images[i])
        # start = time.time()
        predictions = keypoint_detector.predict(image)
        # end = time.time()
        # print(end-start)
        for i in range(len(predictions)):
            cv2.circle(imageg,(predictions[i,0], predictions[i,1]), 3, (0,0,255), -1))
        cv2.imshow("keypoints", image)
        cv2.waitKey(200)
