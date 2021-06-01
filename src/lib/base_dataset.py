import os.path as osp
from torch.utils.data import Dataset
import cv2

from src.lib import transform_cv2 as T


class BaseDataset(Dataset):

    def __init__(self, data_root, ann_path, trans_func=None, mode='train'):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.trans_func = trans_func

        self.lb_map = None

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
        img, label_ = cv2.imread(im_pth)[:, :, ::-1], cv2.imread(lb_pth, 0)
        assert img.shape[:2] == label_.shape[:2], f'image: {im_pth}, label: {lb_pth}\n' \
                                                  f'image shape: {img.shape}, label shape: {label_.shape}'
        if self.lb_map is not None:
            label_ = self.lb_map[label_]
        im_lb = dict(im=img, lb=label_)
        if self.trans_func is not None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label_ = im_lb['im'], im_lb['lb']

        return img.detach(), label_.unsqueeze(0).detach()

    def __len__(self):
        return self.len


class TransformationTrain(object):

    def __init__(self, scales, crop_size):
        self.trans_func = T.Compose([
            T.RandomResizedCrop(scales, crop_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)

        return im_lb


class TransformationVal(object):

    def __init__(self, scales, crop_size):
        self.trans_func = T.Compose([
            T.RandomResizedCrop(scales, crop_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        im, lb = im_lb['im'], im_lb['lb']

        return dict(im=im, lb=lb)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.lib.cityscapes_cv2 import Cityscapes

    ds = Cityscapes(data_root='data/', ann_path='data/val.txt', mode='val')
    dl = DataLoader(ds,
                    batch_size=4,
                    shuffle=True,
                    num_workers=4,
                    drop_last=True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
