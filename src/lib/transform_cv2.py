import math
import numpy as np
import cv2
import torch

from src.lib.consts import MEAN, STD


class RandomResizedCrop(object):
    """
    size should be a tuple of (H, W)
    """

    def __init__(self, scales=(0.5, 1.), size=(384, 384), is_random=True):
        self.scales = scales
        self.size = size
        self.is_random = is_random

    def __call__(self, im_lb):
        if self.size is None:
            return im_lb

        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2], f'image shape is {im.shape}, label shape is {lb.shape}'

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales)) if self.is_random else 1.
        im_h, im_w = [math.ceil(el * scale) for el in self.size]
        im = cv2.resize(im, (im_w, im_h))
        lb = cv2.resize(lb, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        if (im_h, im_w) == (crop_h, crop_w):
            return dict(im=im, lb=lb)

        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            lb = np.pad(lb, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)

        im_h, im_w, _ = im.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))

        return dict(
            im=im[sh:sh + crop_h, sw:sw + crop_w, :].copy(),
            lb=lb[sh:sh + crop_h, sw:sw + crop_w].copy()
        )


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if np.random.random() < self.p:
            return im_lb
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        return dict(
            im=im[:, ::-1, :],
            lb=lb[:, ::-1],
        )


class ColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        if brightness is not None and brightness >= 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if contrast is not None and contrast >= 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if saturation is not None and saturation >= 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        if self.brightness is not None:
            rate = np.random.uniform(*self.brightness)
            im = self.adj_brightness(im, rate)
        if self.contrast is not None:
            rate = np.random.uniform(*self.contrast)
            im = self.adj_contrast(im, rate)
        if self.saturation is not None:
            rate = np.random.uniform(*self.saturation)
            im = self.adj_saturation(im, rate)

        return dict(im=im, lb=lb, )

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


class ImageColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        if brightness is not None and brightness >= 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if contrast is not None and contrast >= 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if saturation is not None and saturation >= 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, image):
        if self.brightness is not None:
            rate = np.random.uniform(*self.brightness)
            image = self.adj_brightness(image, rate)
        if self.contrast is not None:
            rate = np.random.uniform(*self.contrast)
            image = self.adj_contrast(image, rate)
        if self.saturation is not None:
            rate = np.random.uniform(*self.saturation)
            image = self.adj_saturation(image, rate)

        return image

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


class ImageRandomResizeAndCrop(object):
    def __init__(self, scales=(0.5, 1.), size=(384, 384), is_random=True, is_label=False):
        self.scales = scales
        self.size = size
        self.is_random = is_random
        self.is_label = is_label

    def __call__(self, image):
        if self.size is None:
            return image

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales)) if self.is_random else 1.
        im_h, im_w = [math.ceil(el * scale) for el in self.size]
        image = cv2.resize(image, (im_w, im_h))

        if (im_h, im_w) == (crop_h, crop_w):
            return image

        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))

        im_h, im_w, _ = image.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))

        return image[sh:sh + crop_h, sw:sw + crop_w, :]


def label_random_resize_and_crop(image, scales=(0.5, 1.), size=(384, 384), is_random=True, is_label=False):
    if size is None:
        return image

    crop_h, crop_w = size
    scale = np.random.uniform(min(scales), max(scales)) if is_random else 1.
    im_h, im_w = [math.ceil(el * scale) for el in size]
    interpolation = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    image = cv2.resize(image, (im_w, im_h), interpolation=interpolation)

    if (im_h, im_w) == (crop_h, crop_w):
        return image

    pad_h, pad_w = 0, 0
    if im_h < crop_h:
        pad_h = (crop_h - im_h) // 2 + 1
    if im_w < crop_w:
        pad_w = (crop_w - im_w) // 2 + 1
    if pad_h > 0 or pad_w > 0:
        if is_label:
            image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)
        else:
            image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))

    im_h, im_w, _ = image.shape
    sh, sw = np.random.random(2)
    sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))

    if is_label:
        return image[sh:sh + crop_h, sw:sw + crop_w]

    return image[sh:sh + crop_h, sw:sw + crop_w, :]


def label_to_tensor(label):
    if label is not None:
        label = torch.from_numpy(label.astype(np.int64).copy()).clone()

    return label


def image_to_tensor(image, mean=MEAN, std=STD):
    im = image.transpose(2, 0, 1).astype(np.float32)
    im = torch.from_numpy(im).div_(255)
    dtype, device = im.dtype, im.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)[:, None, None]
    std = torch.as_tensor(std, dtype=dtype, device=device)[:, None, None]
    im = im.sub_(mean).div_(std).clone()

    return im


class Compose(object):

    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)

        return im_lb
