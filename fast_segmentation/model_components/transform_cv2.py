import math
import numpy as np
import cv2
import torch
from typing import Tuple, Dict

from fast_segmentation.model_components.consts import MEAN, STD


class RandomResizedCrop(object):
    """
    size should be a tuple of (H, W)
    """

    def __init__(self, size: Tuple[int, int], scales: Tuple = (1.,), is_random: bool = True):
        self.scales = scales
        self.size = size
        self.is_random = is_random

    def __call__(self, im_label):
        if self.size is None:
            return im_label

        image, label = im_label['image'], im_label['label']
        assert image.shape[:2] == label.shape[:2], f'image shape is {image.shape}, label shape is {label.shape}'

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales)) if self.is_random else 1.
        im_h, im_w = [math.ceil(el * scale) for el in self.size]
        image = cv2.resize(image, (im_w, im_h))
        label = cv2.resize(label, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        if (im_h, im_w) == (crop_h, crop_w):
            return dict(image=image, label=label)

        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            label = np.pad(label, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)

        im_h, im_w, _ = image.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))

        return dict(
            image=image[sh:sh + crop_h, sw:sw + crop_w, :].copy(),
            label=label[sh:sh + crop_h, sw:sw + crop_w].copy()
        )


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image_label):
        if np.random.random() < self.p:
            return image_label
        image, label = image_label['image'], image_label['label']
        assert image.shape[:2] == label.shape[:2]
        return dict(
            image=image[:, ::-1, :],
            label=label[:, ::-1],
        )


class ColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None, is_random=True):
        self.is_random = is_random
        if brightness is not None and brightness >= 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if contrast is not None and contrast >= 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if saturation is not None and saturation >= 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, im_label):
        im, label = im_label['image'], im_label['label']
        assert im.shape[:2] == label.shape[:2]
        if self.brightness is not None:
            rate = np.random.uniform(*self.brightness) if self.is_random else np.mean(self.brightness)
            im = self.adj_brightness(im, rate)
        if self.contrast is not None:
            rate = np.random.uniform(*self.contrast) if self.is_random else np.mean(self.contrast)
            im = self.adj_contrast(im, rate)
        if self.saturation is not None:
            rate = np.random.uniform(*self.saturation) if self.is_random else np.mean(self.saturation)
            im = self.adj_saturation(im, rate)

        return dict(image=im, label=label, )

    @staticmethod
    def adj_saturation(im, rate):
        m = np.float32([
            [1 + 2 * rate, 1 - rate, 1 - rate],
            [1 - rate, 1 + 2 * rate, 1 - rate],
            [1 - rate, 1 - rate, 1 + 2 * rate]
        ])
        shape = im.shape
        im = np.matmul(im.reshape(-1, 3), m).reshape(shape) / 3
        im = np.clip(im, 0, 255).astype(np.uint8)
        return im

    @staticmethod
    def adj_brightness(im, rate):
        table = np.array([
            i * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]

    @staticmethod
    def adj_contrast(im, rate):
        table = np.array([
            74 + (i - 74) * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]


class ToTensor(object):
    """Convert numpy arrays in sample to Tensors."""

    def __init__(self, mean=MEAN, std=STD, normalize: bool = True):
        self.mean = mean
        self.std = std
        self.normalize = normalize

    def __call__(self, image_label: Dict[str, np.ndarray]):
        image, label = image_label['image'], image_label['label']
        if label is not None:
            label = torch.from_numpy(label.astype(np.int64).copy()).clone()

        if self.normalize:
            image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
            image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])

            image = image.transpose([2, 0, 1]).astype(np.float32)

            image_std = np.std(image, axis=(1, 2))
            std_division = image_std / self.std
            image = image / std_division[:, None, None]

            image_mean = np.mean(image, axis=(1, 2))
            mean_sub = image_mean - self.mean
            image = image - mean_sub[:, None, None]

        image = torch.as_tensor(image, dtype=torch.float)

        return {'image': image, 'label': label}


class Compose(object):

    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, image_label):
        for comp in self.do_list:
            image_label = comp(image_label)

        return image_label
