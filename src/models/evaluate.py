import os
import os.path as osp
import logging
import argparse
import math
from tabulate import tabulate

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.distributed as dist

from src.configs import cfg_factory
from src.lib.architectures import model_factory
from src.lib.logger import setup_logger
from src.lib.cityscapes_cv2 import get_data_loader
from src.models.consts import IGNORE_LABEL, NUM_CLASSES


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank',
                       type=int, default=-1, )
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='/home/bina/PycharmProjects/tevel-segmentation/models/model_final_0.pth', )
    parse.add_argument('--port', dest='port', type=int, default=44553, )
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2', )
    return parse.parse_args()


args = parse_args()
cfg = cfg_factory[args.model]


class MscEvalV0(object):

    def __init__(self, scales=(0.5,), flip=False, ignore_label=IGNORE_LABEL):
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label

    def __call__(self, net, dl, n_classes):
        # evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            d_iter = enumerate(dl)
        else:
            d_iter = enumerate(tqdm(dl))

        for i, (imgs, labels) in d_iter:
            n, _, h, w = labels.shape
            labels = labels.squeeze(1).cuda()
            size = labels.size()[-2:]
            probs = torch.zeros((n, n_classes, h, w), dtype=torch.float32).cuda().detach()

            for scale in self.scales:
                s_h, s_w = int(scale * h), int(scale * w)
                im_sc = f.interpolate(imgs, size=(s_h, s_w), mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3,))

                logits = net(im_sc)[0]

                if self.flip:
                    logits = torch.flip(logits, dims=(3,))

                logits = f.interpolate(logits, size=size, mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)

            # calc histogram of the predictions in each class
            preds = torch.argmax(probs, dim=1)
            relevant_labels = labels != self.ignore_label
            hist += torch.bincount(labels[relevant_labels] * n_classes + preds[relevant_labels],
                                   minlength=n_classes ** 2).view(n_classes, n_classes)

        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)

        # diagonal is the intersection and the
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag() + 1e-6)
        ious[ious != ious] = 0  # replace nan with zero
        miou = ious.mean()

        return miou.item()


class MscEvalCrop(object):

    def __init__(self, crop_size, crop_stride, flip=True, scales=(0.5, 0.75, 1, 1.25, 1.5, 1.75), lb_ignore=255):

        self.scales = scales
        self.ignore_label = lb_ignore
        self.flip = flip
        self.distributed = dist.is_initialized()

        self.crop_size = crop_size if isinstance(crop_size, (list, tuple)) else (crop_size, crop_size)
        self.crop_stride = crop_stride

    def pad_tensor(self, inten):
        n, c, h, w = inten.size()
        crop_h, crop_w = self.crop_size
        if crop_h < h and crop_w < w:
            return inten, [0, h, 0, w]
        pad_h, pad_w = max(crop_h, h), max(crop_w, w)
        outten = torch.zeros(n, c, pad_h, pad_w).cuda()
        outten.requires_grad_(False)
        margin_h, margin_w = pad_h - h, pad_w - w
        hst, hed = margin_h // 2, margin_h // 2 + h
        wst, wed = margin_w // 2, margin_w // 2 + w
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]

    def eval_chip(self, net, crop):
        prob = net(crop)[0].softmax(dim=1)
        if self.flip:
            crop = torch.flip(crop, dims=(3,))
            prob += net(crop)[0].flip(dims=(3,)).softmax(dim=1)
            prob = torch.exp(prob)
        return prob

    def crop_eval(self, net, im, n_classes):
        crop_h, crop_w = self.crop_size
        stride_rate = self.crop_stride
        im, indices = self.pad_tensor(im)
        n, c, h, w = im.size()

        stride_h = math.ceil(crop_h * stride_rate)
        stride_w = math.ceil(crop_w * stride_rate)
        n_h = math.ceil((h - crop_h) / stride_h) + 1
        n_w = math.ceil((w - crop_w) / stride_w) + 1
        prob = torch.zeros(n, n_classes, h, w).cuda()
        prob.requires_grad_(False)
        for i in range(n_h):
            for j in range(n_w):
                st_h, st_w = stride_h * i, stride_w * j
                end_h, end_w = min(h, st_h + crop_h), min(w, st_w + crop_w)
                st_h, st_w = end_h - crop_h, end_w - crop_w
                chip = im[:, :, st_h:end_h, st_w:end_w]
                prob[:, :, st_h:end_h, st_w:end_w] += self.eval_chip(net, chip)
        hst, hed, wst, wed = indices
        prob = prob[:, :, hst:hed, wst:wed]
        return prob

    def scale_crop_eval(self, net, im, scale, n_classes):
        n, c, h, w = im.size()
        new_hw = [int(h * scale), int(w * scale)]
        im = f.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(net, im, n_classes)
        prob = f.interpolate(prob, (h, w), mode='bilinear', align_corners=True)
        return prob

    @torch.no_grad()
    def __call__(self, net, dl, n_classes):
        dloader = dl if self.distributed and not dist.get_rank() == 0 else tqdm(dl)

        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        hist.requires_grad_(False)

        for i, (imgs, label) in enumerate(dloader):
            imgs = imgs.cuda()
            label = label.squeeze(1).cuda()
            n, h, w = label.shape
            probs = torch.zeros((n, n_classes, h, w)).cuda()
            probs.requires_grad_(False)
            for sc in self.scales:
                probs += self.scale_crop_eval(net, imgs, sc, n_classes)
            torch.cuda.empty_cache()
            preds = torch.argmax(probs, dim=1)

            keep = label != self.ignore_label
            hist += torch.bincount(label[keep] * n_classes + preds[keep], minlength=n_classes ** 2). \
                view(n_classes, n_classes)

        if self.distributed:
            dist.all_reduce(hist, dist.ReduceOp.SUM)

        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        ious[ious != ious] = 0  # replace nan with zero
        miou = ious.mean()
        return miou.item()


@torch.no_grad()
def eval_model(net, ims_per_gpu, im_root, im_anns):
    is_dist = dist.is_initialized()
    dl = get_data_loader(im_root, im_anns, ims_per_gpu, cfg.scales, cfg.crop_size, mode='val', distributed=is_dist)
    net.eval()

    heads, mious = [], []
    logger = logging.getLogger()

    single_scale = MscEvalV0((1.,), False)
    miou = single_scale(net, dl, NUM_CLASSES)
    heads.append('single_scale')
    mious.append(miou)
    logger.info('single mIOU is: %s\n', miou)

    single_crop = MscEvalCrop(crop_size=cfg.crop_size, crop_stride=2. / 3, flip=False, scales=[1.],
                              lb_ignore=IGNORE_LABEL)
    miou = single_crop(net, dl, NUM_CLASSES)
    heads.append('single_scale_crop')
    mious.append(miou)
    logger.info('single scale crop mIOU is: %s\n', miou)

    ms_flip = MscEvalV0((0.5, 0.75, 1, 1.25, 1.5, 1.75), True)
    miou = ms_flip(net, dl, NUM_CLASSES)
    heads.append('ms_flip')
    mious.append(miou)
    logger.info('ms flip mIOU is: %s\n', miou)

    ms_flip_crop = MscEvalCrop(crop_size=cfg.crop_size, crop_stride=2. / 3, flip=True,
                               scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75], lb_ignore=IGNORE_LABEL)
    miou = ms_flip_crop(net, dl, NUM_CLASSES)
    heads.append('ms_flip_crop')
    mious.append(miou)
    logger.info('ms crop mIOU is: %s\n', miou)
    return heads, mious


def evaluate(cfg_, weight_pth):
    logger = logging.getLogger()

    # model
    logger.info('setup and restore model')
    net = model_factory[cfg_.model_type](NUM_CLASSES)
    net.load_state_dict(torch.load(weight_pth))
    net.cuda()

    is_dist = dist.is_initialized()
    if is_dist:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank, ], output_device=local_rank)

    # evaluator
    heads, mious = eval_model(net, cfg_.ims_per_gpu, cfg_.im_root, cfg_.val_im_anns)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))


def main():
    if not args.local_rank == -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:{}'.format(args.port),
                                world_size=torch.cuda.device_count(),
                                rank=args.local_rank
                                )
    if not osp.exists(cfg.respth):
        os.makedirs(cfg.respth)
    setup_logger('{}-eval'.format(cfg.model_type), cfg.respth)
    evaluate(cfg, args.weight_pth)


if __name__ == "__main__":
    main()
