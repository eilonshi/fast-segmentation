#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys

sys.path.insert(0, '')
import os
import os.path as osp
import logging
import argparse
import math
from tabulate import tabulate

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from src.tools.consts import NUM_CLASSES
from src.configs import cfg_factory
from src.lib.models import model_factory
from src.lib.logger import setup_logger
from src.lib.cityscapes_cv2 import get_data_loader


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank',
                       type=int, default=-1, )
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='model_final.pth', )
    parse.add_argument('--port', dest='port', type=int, default=44553, )
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2', )
    return parse.parse_args()


args = parse_args()
cfg = cfg_factory[args.model]


class MscEvalV0(object):

    def __init__(self, scales=(0.5,), flip=False, ignore_label=255):
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

        for i, (imgs, label) in d_iter:
            n, _, h, w = label.shape
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            probs = torch.zeros((n, n_classes, h, w), dtype=torch.float32).cuda().detach()

            for scale in self.scales:
                s_h, s_w = int(scale * h), int(scale * w)
                im_sc = F.interpolate(imgs, size=(s_h, s_w), mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                logits = net(im_sc)[0]
                logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3,))
                    logits = net(im_sc)[0]
                    logits = torch.flip(logits, dims=(3,))
                    logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
            ).view(n_classes, n_classes)
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()


class MscEvalCrop(object):

    def __init__(
            self,
            crop_size=1024,
            crop_stride=2. / 3,
            flip=True,
            scales=(0.5, 0.75, 1, 1.25, 1.5, 1.75),
            lb_ignore=255,
    ):
        self.scales = scales
        self.ignore_label = lb_ignore
        self.flip = flip
        self.distributed = dist.is_initialized()

        self.crop_size = crop_size if isinstance(crop_size, (list, tuple)) else (crop_size, crop_size)
        self.crop_stride = crop_stride

    def pad_tensor(self, inten):
        N, C, H, W = inten.size()
        cropH, cropW = self.crop_size
        if cropH < H and cropW < W: return inten, [0, H, 0, W]
        padH, padW = max(cropH, H), max(cropW, W)
        outten = torch.zeros(N, C, padH, padW).cuda()
        outten.requires_grad_(False)
        marginH, marginW = padH - H, padW - W
        hst, hed = marginH // 2, marginH // 2 + H
        wst, wed = marginW // 2, marginW // 2 + W
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
                stH, stW = stride_h * i, stride_w * j
                endH, endW = min(h, stH + crop_h), min(w, stW + crop_w)
                stH, stW = endH - crop_h, endW - crop_w
                chip = im[:, :, stH:endH, stW:endW]
                prob[:, :, stH:endH, stW:endW] += self.eval_chip(net, chip)
        hst, hed, wst, wed = indices
        prob = prob[:, :, hst:hed, wst:wed]
        return prob

    def scale_crop_eval(self, net, im, scale, n_classes):
        N, C, H, W = im.size()
        new_hw = [int(H * scale), int(W * scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(net, im, n_classes)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
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
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
            ).view(n_classes, n_classes)

        if self.distributed:
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
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

    single_crop = MscEvalCrop(
        crop_size=1024,
        crop_stride=2. / 3,
        flip=False,
        scales=[1.],
        lb_ignore=255,
    )
    miou = single_crop(net, dl, NUM_CLASSES)
    heads.append('single_scale_crop')
    mious.append(miou)
    logger.info('single scale crop mIOU is: %s\n', miou)

    ms_flip = MscEvalV0((0.5, 0.75, 1, 1.25, 1.5, 1.75), True)
    miou = ms_flip(net, dl, NUM_CLASSES)
    heads.append('ms_flip')
    mious.append(miou)
    logger.info('ms flip mIOU is: %s\n', miou)

    ms_flip_crop = MscEvalCrop(
        crop_size=1024,
        crop_stride=2. / 3,
        flip=True,
        scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        lb_ignore=255,
    )
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
    #  net = BiSeNetV2(NUM_CLASSES)
    net.load_state_dict(torch.load(weight_pth))
    net.cuda()

    is_dist = dist.is_initialized()
    if is_dist:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank
        )

    # evaluator
    heads, mious = eval_model(net, 2, cfg_.im_root, cfg_.val_im_anns)
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
