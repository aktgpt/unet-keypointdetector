import glob, os
import json
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import random
import cv2
import numpy as np
import torch
import matplotlib.pylab as plt
from imgaug import augmenters as iaa
import imgaug as ia
from torchvision.utils import make_grid


class MultipleMarkersDataset(Dataset):
    def __init__(self, config):
        self.image_folder = config['data']['image_folder']
        self.annotation_folder = config['data']['annotation_folder']
        self.augment = config['data']['augment']
        self.unet_image_size = config['unet']['input_shape']

        self.kernel_size = config['mask']['kernel_size']
        self.variance =  config['mask']['variance']

        self.annotations = list()
        search_path = os.path.join(self.annotation_folder, "*.json")
        for file_path in glob.glob(search_path):
            with open(file_path) as json_file:
                annot = json.load(json_file)
                image_file = os.path.join(self.image_folder, annot['file_name'])
                image_shape = annot['image_shape']
                for i in range(len(annot['object'])):
                    self.annotations.append({
                        'image_file': image_file,
                        'image_shape': image_shape,
                        'object': annot['object'][i]
                    })

        bbox_scale_factor = config['data']['bbox_scale_factor']
        if bbox_scale_factor:
            self.modify_bbox_annotations(bbox_scale_factor)

        negative_sample_ratio = config['data']['negative_sample_ratio']
        if negative_sample_ratio:
            self.add_negative_images(negative_sample_ratio)

        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        self.transform = iaa.Sequential(
            [
                iaa.Fliplr(0.2),
                iaa.Flipud(0.2),
                sometimes(iaa.Crop(percent=(0, 0.1))),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-5, 5),
                    shear=(-5, 5),
                    order=[0, 1],
                )),

                iaa.SomeOf((0, 5),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200), name="superpixels")),
                                   iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0), name="gaussianblur"),
                                   iaa.AverageBlur(k=(2, 7), name="averageblur"),
                                   iaa.MedianBlur(k=(3, 11), name="medianblur"),
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5), name="sharpen"),
                               # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                               # sometimes(iaa.OneOf([
                               #     iaa.EdgeDetect(alpha=(0, 0.7)),
                               #     iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               # ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5, name="additivegaussianblur"),
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Invert(0.05, per_channel=True, name="invertchannel"),
                               iaa.Add((-10, 10), per_channel=0.5, name="colorjitter"),
                               iaa.Multiply((0.5, 1.5), per_channel=0.5, name="changebrightness"),
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5, name="contrastnorm"),  # improve or worsen the contrast
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25, name="elastictransform")),
                               # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        self.input_only_transform = ['superpixels', 'gaussianblur', 'averageblur', 'medianblur', 'sharpen',
                                     'additivegaussianblur', 'invertchannel', 'colorjitter', 'changebrightness',
                                     'contrastnorm', 'elastictransform']

    def add_negative_images(self, negative_sample_ratio):
        len_dataset = len(self.annotations)
        sample_factor = int(1/ negative_sample_ratio)
        for i in range(len_dataset):
            if (i+1) % sample_factor == 0:
                annotation_new = {}
                bbox = self.annotations[i]['object']['bbox_dimensions'][0]
                image_shape = self.annotations[i]['image_shape']
                annotation_new['image_file'] = self.annotations[i]['image_file']
                annotation_new['image_shape'] = self.annotations[i]['image_shape']
                annotation_new['object'] = {}
                bbox_new = self.get_random_bbox(bbox, image_shape)
                if bbox_new is not []:
                    annotation_new['object']['bbox_dimensions'] = bbox_new
                    annotation_new['object']['points_location'] = [[]]
                    self.annotations.append(annotation_new)
                else:
                    pass
            else:
                pass

    @staticmethod
    def get_random_bbox(bbox, image_shape):
        xmin, xmax, ymin, ymax = bbox
        yshape, xshape = image_shape[:2]
        bbox_x = xmax - xmin
        bbox_y = ymax - ymin
        xrange = []
        if xmin-bbox_x > 0:
            xrange.append(list(range(0, xmin-bbox_x)))
        if xshape-bbox_x > xmax:
            xrange.append(list(range(xmax, xshape-bbox_x)))
        yrange = []
        if ymin-bbox_y > 0:
            yrange.append(list(range(0, ymin-bbox_y)))
        if yshape-bbox_y > ymax:
            yrange.append(list(range(ymax, yshape-bbox_y)))
        if xrange is not [] and yrange is not []:
            xmin_new = random.choice(random.choice(xrange))
            ymin_new = random.choice(random.choice(yrange))
            xmax_new = xmin_new + bbox_x
            ymax_new = ymin_new + bbox_y
            return [[xmin_new, xmax_new, ymin_new, ymax_new]]
        else:
            return []

    def modify_bbox_annotations(self, bbox_scale_factor):
        annotations_new = []
        for annotation in self.annotations:
            annotation_new = dict()
            annotation_new['image_file'] = annotation['image_file']
            annotation_new['image_shape'] = annotation['image_shape']
            annotation_new['object'] = {}
            imgy, imgx = annotation['image_shape'][:2]
            xmin, xmax, ymin, ymax = annotation['object']['bbox_dimensions'][0]
            pixels_add_x = int((xmax-xmin)*bbox_scale_factor/2)
            pixels_add_y = int((ymax-ymin)*bbox_scale_factor/2)
            xmin_n = max(0, xmin - pixels_add_x)
            ymin_n = max(0, ymin - pixels_add_y)
            xmax_n = min(imgx, xmax + pixels_add_x)
            ymax_n = min(imgy, ymax + pixels_add_y)
            annotation_new['object']['bbox_dimensions'] = [[xmin_n, xmax_n, ymin_n, ymax_n]]
            annotation_new['object']['points_location'] = annotation['object']['points_location']
            annotations_new.append(annotation_new)
        self.annotations = annotations_new

    def __len__(self):
        return len(self.annotations)

    def _activator_masks(self, images, augmenter, parents, default):
        if self.input_only_transform and augmenter.name in self.input_only_transform:
            return False
        else:
            return default

    def __getitem__(self, index):
        annotation = self.annotations[index]
        image = self.extract_image(annotation)
        mask = self.extract_mask(annotation)
        if self.augment:
            det_tf = self.transform.to_deterministic()
            image = det_tf.augment_image(image)
            mask = det_tf.augment_image(mask, hooks=ia.HooksImages(activator=self._activator_masks))

        to_tensor = ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask)

        # plt.imshow(mask[0,:,:])
        # plt.show()
        # opencvimage = np.hstack((cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
        # cv2.imshow('Main', opencvimage)
        # cv2.waitKey(10)

        sample = {"image": image,
                  "mask": mask}

        return sample


    def extract_image(self, annotation):
        img_name = annotation['image_file']
        xmin, xmax, ymin, ymax = self.get_bbox(annotation)
        if self.unet_image_size[2] == 1:
            image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(img_name)

        image = image[ymin:ymax, xmin:xmax]
        train_image = cv2.resize(image, (self.unet_image_size[1], self.unet_image_size[0]))
        return train_image

    def get_bbox(self, annotation):
        xmin = annotation['object']['bbox_dimensions'][0][0]
        xmax = annotation['object']['bbox_dimensions'][0][1]
        ymin = annotation['object']['bbox_dimensions'][0][2]
        ymax = annotation['object']['bbox_dimensions'][0][3]
        return xmin, xmax, ymin, ymax

    def get_point_locations(self, annotation):
        xmin, xmax, ymin, ymax = self.get_bbox(annotation)
        rx = self.unet_image_size[0] / (xmax-xmin)
        ry = self.unet_image_size[1] / (ymax-ymin)
        px = []
        py = []
        for i in range(len(annotation['object']['points_location'])):
            if annotation['object']['points_location'][i] == []:
                pass
            else:
                px.append(int((annotation['object']['points_location'][i][0] - xmin)*rx))
                py.append(int((annotation['object']['points_location'][i][1] - ymin)*ry))
        return px, py

    def extract_mask(self, annotation):
        px, py = self.get_point_locations(annotation)

        if px == [] or py == []:

            mask = np.zeros((self.unet_image_size[0], self.unet_image_size[1]))
            # mask = np.array(mask).astype(np.float32)
        else:
            gauss_kernel = self.gaussian_2d((self.kernel_size, self.kernel_size), self.variance)
            mask = np.zeros((self.unet_image_size[0], self.unet_image_size[1]))
            for i in range(len(px)):
                ex, ey = [px[i] - self.kernel_size // 2, py[i] - self.kernel_size // 2]
                v_range1 = slice(max(0, ey), max(min(ey + gauss_kernel.shape[0], self.unet_image_size[0]), 0))
                h_range1 = slice(max(0, ex), max(min(ex + gauss_kernel.shape[1], self.unet_image_size[1]), 0))
                v_range2 = slice(max(0, -ey), min(-ey + self.unet_image_size[0], gauss_kernel.shape[0]))
                h_range2 = slice(max(0, -ex), min(-ex + self.unet_image_size[1], gauss_kernel.shape[1]))
                try:
                    mask[v_range1, h_range1] += gauss_kernel[v_range2, h_range2]
                except:
                    print("error")
        # plt.imshow(mask)
        # plt.show()
        mask = mask[..., np.newaxis]
        return mask

    @staticmethod
    def gaussian_2d(shape=(3, 3), sigma=0.5):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_copy = np.flip(image.transpose((2, 0, 1)), axis=0).copy()
        image_torch = torch.from_numpy(image_copy).float().to(device)

        eps = 1 / (image.shape[0]*image.shape[1])
        image_torch = torch.max(image_torch, torch.cuda.FloatTensor([eps]))

        return image_torch


def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, masks_batch = sample_batched['image'].numpy().astype(np.uint8), sample_batched['mask'].numpy().astype(np.bool)
    batch_size = len(images_batch)
    for i in range(batch_size):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.tight_layout()
        plt.imshow(images_batch[i].transpose((1, 2, 0)))
        plt.subplot(1, 2, 2)
        plt.tight_layout()
    plt.imshow(np.squeeze(masks_batch[i].transpose((1, 2, 0))))


def validation_split(dataset, validation_ratio):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_ratio * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, validation_sampler


def show(img):
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


def get_data_loaders(dataset_config):
    dataset = MultipleMarkersDataset(dataset_config)
    
    # imgs = [dataset[i] for i in range(6)]
    # show(make_grid(torch.stack([img['image'].cpu() for img in imgs])))
    # show(make_grid(torch.stack([img['mask'].cpu() for img in imgs])))

    print("Images found for training:" + str(len(dataset)))

    train_sampler, validation_sampler = validation_split(dataset, dataset_config['train']['val_split_ratio'])
    print('{} images found in training set'.format(len(train_sampler)))
    print('{} images found in validation set'.format(len(validation_sampler)))
    train_loader = DataLoader(dataset, batch_size=dataset_config['train']['batch_size'], sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=dataset_config['train']['batch_size'], sampler=validation_sampler)

    # for i_batch, sample_batched in enumerate(train_loader):
    #     show_batch(sample_batched)
    #     plt.show()

    return train_loader, validation_loader
