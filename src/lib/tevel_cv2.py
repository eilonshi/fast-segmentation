import os.path as osp

import cv2
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from src.lib import transform_cv2 as t
from src.lib.consts import STD, MEAN

from src.lib.sampler import RepeatedDistSampler
from src.lib.transform_cv2 import image_to_tensor, label_to_tensor
from src.models.consts import NUM_CLASSES, NUM_WORKERS


class TevelDataset(Dataset):
    def __init__(self, data_root, ann_path, trans_func=None, mode='train'):
        super(TevelDataset, self).__init__()
        assert mode in ('train', 'val')

        self.n_cats = NUM_CLASSES
        self.lb_ignore = 255
        self.mode = mode
        self.trans_func = trans_func

        with open(ann_path, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []

        for pair in pairs:
            img_pth, lb_pth = pair.split(',')
            self.img_paths.append(osp.join(data_root, img_pth))
            self.lb_paths.append(osp.join(data_root, lb_pth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)

    def __getitem__(self, idx):
        im_pth, lb_pth = self.img_paths[idx], self.lb_paths[idx]
        assert cv2.imread(im_pth) is not None, im_pth
        assert cv2.imread(lb_pth, 0) is not None, lb_pth

        img, label_ = cv2.cvtColor(cv2.imread(im_pth), cv2.COLOR_BGR2RGB), cv2.imread(lb_pth, 0)
        assert img.shape[:2] == label_.shape[:2], f'image: {im_pth}, label: {lb_pth}\n' \
                                                  f'image shape: {img.shape}, label shape: {label_.shape}'

        im_lb = dict(im=img, lb=label_)
        if self.trans_func is not None:
            im_lb = self.trans_func(im_lb)

        img_tensor = image_to_tensor(im_lb['im'], mean=MEAN, std=STD)
        label_tensor = label_to_tensor(im_lb['lb'])

        return img_tensor.detach(), label_tensor.unsqueeze(0).detach()

    def __len__(self):
        return self.len


class TransformationTrain(object):

    def __init__(self, scales, crop_size):
        self.trans_func = t.Compose([
            t.RandomResizedCrop(scales, crop_size),
            t.RandomHorizontalFlip(),
            t.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)

        return im_lb


class TransformationVal(object):

    def __init__(self, crop_size):
        self.trans_func = t.Compose([
            t.RandomResizedCrop(size=crop_size, is_random=False),
            t.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        ])

    def __call__(self, im_lb):
        return self.trans_func(im_lb)


class TransformationInference(object):

    def __init__(self, crop_size):
        self.trans_func = t.Compose([
            t.ImageRandomResizeAndCrop(size=crop_size, is_random=False),
            t.ImageColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        ])

    def __call__(self, im_lb):
        return self.trans_func(im_lb)


def get_data_loader(data_path, ann_path, ims_per_gpu, scales, crop_size, max_iter=None, mode='train', distributed=True):
    if mode == 'train':
        trans_func = TransformationTrain(scales, crop_size)
        batch_size = ims_per_gpu
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = TransformationVal(crop_size)
        batch_size = ims_per_gpu
        shuffle = False
        drop_last = False
    else:
        raise ValueError

    ds_ = TevelDataset(data_path, ann_path, trans_func=trans_func, mode=mode)

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


if __name__ == "__main__":
    ds = TevelDataset(data_root='./data/', ann_path='./data/val.txt', mode='val')
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    for images, label in dl:
        print(len(images))
        for el in images:
            print(el.size())
        break
