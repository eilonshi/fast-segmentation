import os.path as osp

import cv2
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from fast_segmentation.model_components import transform_cv2 as t
from fast_segmentation.model_components.sampler import RepeatedDistSampler
from fast_segmentation.model_components.transform_cv2 import ToTensor
from fast_segmentation.core.consts import NUM_CLASSES, NUM_WORKERS, IGNORE_LABEL


class UrbanDataset(Dataset):
    def __init__(self, data_root: str, ann_path: str, trans_func: callable = None, mode: str = 'train'):
        super(UrbanDataset, self).__init__()
        assert mode in ('train', 'val')

        self.n_cats = NUM_CLASSES
        self.label_ignore = IGNORE_LABEL
        self.mode = mode
        self.trans_func = trans_func
        self.to_tensor = ToTensor()

        with open(ann_path, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.label_paths = [], []

        for pair in pairs:
            img_pth, label_pth = pair.split(',')
            self.img_paths.append(osp.join(data_root, img_pth))
            self.label_paths.append(osp.join(data_root, label_pth))

        assert len(self.img_paths) == len(self.label_paths)
        self.len = len(self.img_paths)

    def __getitem__(self, idx):
        im_pth, label_pth = self.img_paths[idx], self.label_paths[idx]
        assert cv2.imread(im_pth) is not None, im_pth
        assert cv2.imread(label_pth, 0) is not None, label_pth

        img, label_ = cv2.cvtColor(cv2.imread(im_pth), cv2.COLOR_BGR2RGB), cv2.imread(label_pth, 0)
        assert img.shape[:2] == label_.shape[:2], f'image: {im_pth}, label: {label_pth}\n' \
                                                  f'image shape: {img.shape}, label shape: {label_.shape}'

        image_label = dict(image=img, label=label_)
        if self.trans_func is not None:
            image_label = self.trans_func(image_label)

        image_label = self.to_tensor(image_label)
        img_tensor, label_tensor = image_label['image'], image_label['label']

        return img_tensor.detach(), label_tensor.unsqueeze(0).detach()

    def __len__(self):
        return self.len


class TransformationTrain(object):

    def __init__(self, scales: Tuple, crop_size: Tuple[int, int]):
        self.trans_func = t.Compose([
            t.RandomResizedCrop(scales=scales, size=crop_size),
            t.RandomHorizontalFlip(),
            t.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        ])

    def __call__(self, image_label):
        image_label = self.trans_func(image_label)

        return image_label


class TransformationVal(object):

    def __init__(self, crop_size):
        self.trans_func = t.Compose([
            t.RandomResizedCrop(size=crop_size, is_random=False),
            t.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, is_random=False)
        ])

    def __call__(self, image_label):
        return self.trans_func(image_label)


def get_data_loader(data_path: str, ann_path: str, ims_per_gpu: int, crop_size: Tuple[int, int], scales: Tuple = None,
                    max_iter: int = None, mode: str = 'train', distributed=True):
    if mode == 'train':
        trans_func = TransformationTrain(scales=scales, crop_size=crop_size)
        batch_size = ims_per_gpu
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = TransformationVal(crop_size=crop_size)
        batch_size = ims_per_gpu
        shuffle = False
        drop_last = False
    else:
        raise ValueError

    ds_ = UrbanDataset(data_path, ann_path, trans_func=trans_func, mode=mode)

    if distributed:
        assert dist.is_available(), "dist should be initialized"
        if mode == 'train':
            assert max_iter is not None
            n_train_images = ims_per_gpu * dist.get_world_size() * max_iter
            sampler = RepeatedDistSampler(ds_, n_train_images, shuffle=shuffle, data_source=None)
        else:
            sampler = RepeatedDistSampler(ds_, ims_per_gpu, shuffle=shuffle, data_source=None)
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=drop_last)
        dl_ = DataLoader(ds_, batch_sampler=batch_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        dl_ = DataLoader(ds_, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=NUM_WORKERS,
                         pin_memory=True)
    return dl_
