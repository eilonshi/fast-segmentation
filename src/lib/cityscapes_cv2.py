import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

from src.lib import transform_cv2 as t
from src.lib.sampler import RepeatedDistSampler
from src.lib.base_dataset import BaseDataset, TransformationTrain, TransformationVal
from src.models.consts import NUM_CLASSES, NUM_WORKERS


class Cityscapes(BaseDataset):
    def __init__(self, data_root, ann_path, trans_func=None, mode='train'):
        super(Cityscapes, self).__init__(data_root, ann_path, trans_func, mode)
        self.n_cats = NUM_CLASSES
        self.lb_ignore = 255

        self.to_tensor = t.ToTensor(
            mean=(0.3257, 0.3690, 0.3223),  # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )


def get_data_loader(data_path, ann_path, ims_per_gpu, scales, crop_size, max_iter=None, mode='train', distributed=True):
    if mode == 'train':
        trans_func = TransformationTrain(scales, crop_size)
        batch_size = ims_per_gpu
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = TransformationVal(scales, crop_size)
        batch_size = ims_per_gpu
        shuffle = False
        drop_last = False
    else:
        raise ValueError

    ds_ = Cityscapes(data_path, ann_path, trans_func=trans_func, mode=mode)

    if distributed:
        assert dist.is_available(), "dist should be initialized"
        if mode == 'train':
            assert max_iter is not None
            n_train_images = ims_per_gpu * dist.get_world_size() * max_iter
            sampler = RepeatedDistSampler(ds_, n_train_images, shuffle=shuffle, data_source=None)
        else:
            # sampler = torch.utils.data.distributed.DistributedSampler(ds_, shuffle=shuffle)
            sampler = RepeatedDistSampler(ds_, ims_per_gpu, shuffle=shuffle, data_source=None)
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=drop_last)
        dl_ = DataLoader(ds_, batch_sampler=batch_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        dl_ = DataLoader(ds_, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=NUM_WORKERS,
                         pin_memory=True)
    return dl_


if __name__ == "__main__":
    ds = Cityscapes(data_root='./data/', ann_path='./data/val.txt', mode='val')
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    for images, label in dl:
        print(len(images))
        for el in images:
            print(el.size())
        break
