import os.path as osp
from torch.utils.data import Dataset
import cv2
from src.lib import transform_cv2 as t


class BaseDataset(Dataset):

    def __init__(self, data_root, ann_path, trans_func=None, mode='train'):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'inference')
        self.mode = mode
        self.trans_func = trans_func

        self.to_tensor = t.ToTensor(
            mean=(0.3257, 0.3690, 0.3223),  # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )

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

        im_lb = dict(im=img, lb=label_)
        if self.trans_func is not None:
            im_lb = self.trans_func(im_lb)

        im_lb_tensor = self.to_tensor(im_lb)
        img_tensor, label_tensor = im_lb_tensor['im'], im_lb_tensor['lb']

        return img_tensor.detach(), label_tensor.unsqueeze(0).detach()

    def __len__(self):
        return self.len


class TransformationTrain(object):

    def __init__(self, scales, crop_size):
        self.trans_func = t.Compose([
            t.RandomResizedCrop(scales, crop_size),
            t.RandomHorizontalFlip(),
            t.ColorJitter(
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
        self.trans_func = t.Compose([
            t.RandomResizedCrop(scales, crop_size, is_random=False),
            t.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        ])

    def __call__(self, im_lb):
        return self.trans_func(im_lb)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.lib.tevel_cv2 import TevelDataset

    ds = TevelDataset(data_root='data/', ann_path='data/val.txt', mode='val')
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
