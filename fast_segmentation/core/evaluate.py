import os
import os.path as osp
import logging
import argparse
import math

import yaml
from tabulate import tabulate
from tqdm import tqdm
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.distributed as dist

from fast_segmentation.model_components.architectures import model_factory
from fast_segmentation.model_components.data_cv2 import get_data_loader
from fast_segmentation.model_components.logger import setup_logger
from fast_segmentation.core.consts import IGNORE_LABEL, NUM_CLASSES, BAD_IOU


def parse_args():
    """
    Creates the parser for evaluation arguments

    Returns:
        The parser
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank',
                       type=int, default=-1)
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='/home/bina/PycharmProjects/fast-segmentation/models/5/best_model.pth')
    parse.add_argument('--im_root', type=str, default='/home/bina/PycharmProjects/fast-segmentation/data')
    parse.add_argument('--val_im_anns', type=str,
                       default='/home/bina/PycharmProjects/fast-segmentation/data/train_small.txt')
    parse.add_argument('--false_analysis_path', type=str,
                       default='/home/bina/PycharmProjects/fast-segmentation/data/false_analysis')
    parse.add_argument('--log_path', type=str,
                       default='/home/bina/PycharmProjects/fast-segmentation/logs/regular_logs')
    parse.add_argument('--port', dest='port', type=int, default=44553, )
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2')
    parse.add_argument('--config_path', type=str,
                       default='/home/bina/PycharmProjects/fast-segmentation/configs/main_cfg.yaml')

    return parse.parse_args()


class MscEvalV0(object):
    """

    """

    def __init__(self, scales=(1.,), flip=False, ignore_label=IGNORE_LABEL):
        self.scales = scales
        self.flip = flip
        self.ignore_label = ignore_label

    def __call__(self, net: nn.Module, data_loader, num_classes):
        # evaluate
        hist = torch.zeros(num_classes, num_classes).cuda().detach()

        if dist.is_initialized() and dist.get_rank() != 0:
            d_iter = enumerate(data_loader)
        else:
            d_iter = enumerate(tqdm(data_loader))

        for i, (imgs, labels) in d_iter:
            n, _, h, w = labels.shape
            labels = labels.squeeze(1).cuda()
            size = labels.size()[-2:]
            probs = torch.zeros((n, num_classes, h, w), dtype=torch.float32).cuda().detach()

            for scale in self.scales:
                s_h, s_w = int(scale * h), int(scale * w)
                im_sc = functional.interpolate(imgs, size=(s_h, s_w), mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3,))

                logits = net(im_sc)[0]

                if self.flip:
                    logits = torch.flip(logits, dims=(3,))

                logits = functional.interpolate(logits, size=size, mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)

            # calc histogram of the predictions in each class
            preds = torch.argmax(probs, dim=1)
            relevant_labels = labels != self.ignore_label
            hist += torch.bincount(labels[relevant_labels] * num_classes + preds[relevant_labels],
                                   minlength=num_classes ** 2).view(num_classes, num_classes)

        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)

        # diagonal is the intersection and the
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag() + 1e-6)
        ious[ious != ious] = 0  # replace nan with zero
        miou = ious.mean()

        return miou.item()


class MscEvalCrop(object):

    def __init__(self, crop_size, crop_stride, false_analysis_path, flip=True, scales=(0.5, 0.75, 1, 1.25, 1.5, 1.75),
                 label_ignore=255):

        self.scales = scales
        self.ignore_label = label_ignore
        self.flip = flip
        self.distributed = dist.is_initialized()

        self.crop_size = crop_size if isinstance(crop_size, (list, tuple)) else (crop_size, crop_size)
        self.crop_stride = crop_stride

        self.false_analysis_path = false_analysis_path

    def pad_tensor(self, in_tensor):
        n, c, h, w = in_tensor.size()
        crop_h, crop_w = self.crop_size

        if crop_h < h and crop_w < w:
            return in_tensor, [0, h, 0, w]

        pad_h, pad_w = max(crop_h, h), max(crop_w, w)
        out_tensor = torch.zeros(n, c, pad_h, pad_w).cuda()
        out_tensor.requires_grad_(False)

        margin_h, margin_w = pad_h - h, pad_w - w
        hst, hed = margin_h // 2, margin_h // 2 + h
        wst, wed = margin_w // 2, margin_w // 2 + w
        out_tensor[:, :, hst:hed, wst:wed] = in_tensor

        return out_tensor, [hst, hed, wst, wed]

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
        im = functional.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(net, im, n_classes)
        prob = functional.interpolate(prob, (h, w), mode='bilinear', align_corners=True)

        return prob

    @torch.no_grad()
    def __call__(self, net, dl, n_classes):
        data_loader = dl if self.distributed and not dist.get_rank() == 0 else tqdm(dl)

        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        hist.requires_grad_(False)

        for i, (imgs, label) in enumerate(data_loader):
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
            cur_hist = torch.zeros(n_classes, n_classes).cuda().detach()

            bin_count = torch.bincount(label[keep] * n_classes + preds[keep], minlength=n_classes ** 2). \
                view(n_classes, n_classes)
            cur_hist += bin_count
            cur_miou = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
            cur_miou[cur_miou != cur_miou] = 0  # replace nan with zero
            cur_miou = cur_miou.mean()

            if cur_miou < BAD_IOU:
                save_in_false_analysis(imgs, self.false_analysis_path)

            hist += bin_count

        if self.distributed:
            dist.all_reduce(hist, dist.ReduceOp.SUM)

        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        ious[ious != ious] = 0  # replace nan with zero
        miou = ious.mean()

        return miou.item()


def save_in_false_analysis(images: torch.Tensor, path: str):
    # TODO: write again this function
    pass

    # images_channels_last = images.permute([0, 2, 3, 1])
    #
    # for i, image in enumerate(images_channels_last):
    #     image = image.detach().cpu().numpy()
    #     file_name = os.path.join(path, f'img{i}.jpg')
    #     print('filename', file_name)
    #     save_labels_mask_with_legend(mask=image, save_path=file_name)


@torch.no_grad()
def eval_model(net: nn.Module, ims_per_gpu: int, crop_size: Tuple[int, int], im_root: str, im_anns: str,
               false_analysis_path: str) -> Tuple[List[str], List[float]]:
    is_dist = dist.is_initialized()
    dl = get_data_loader(data_path=im_root, ann_path=im_anns, ims_per_gpu=ims_per_gpu, crop_size=crop_size, mode='val',
                         distributed=is_dist)
    net.eval()

    heads, mious = [], []
    logger = logging.getLogger()

    single_scale = MscEvalV0((1.,), False)
    miou = single_scale(net, dl, NUM_CLASSES)
    heads.append('single_scale')
    mious.append(miou)
    logger.info('single mIOU is: %s\n', miou)

    single_crop = MscEvalCrop(crop_size=crop_size, crop_stride=2. / 3, flip=False, scales=[1.],
                              label_ignore=IGNORE_LABEL, false_analysis_path=false_analysis_path)
    miou = single_crop(net, dl, NUM_CLASSES)
    heads.append('single_scale_crop')
    mious.append(miou)
    logger.info('single scale crop mIOU is: %s\n', miou)

    ms_flip = MscEvalV0((0.5, 0.75, 1, 1.25, 1.5, 1.75), True)
    miou = ms_flip(net, dl, NUM_CLASSES)
    heads.append('ms_flip')
    mious.append(miou)
    logger.info('ms flip mIOU is: %s\n', miou)

    ms_flip_crop = MscEvalCrop(crop_size=crop_size, crop_stride=2. / 3, flip=True,
                               scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75], label_ignore=IGNORE_LABEL,
                               false_analysis_path=false_analysis_path)
    miou = ms_flip_crop(net, dl, NUM_CLASSES)
    heads.append('ms_flip_crop')
    mious.append(miou)
    logger.info('ms crop mIOU is: %s\n', miou)

    return heads, mious


def evaluate(ims_per_gpu, crop_size, weight_pth, model_type, im_root, val_im_anns, false_analysis_path):
    logger = logging.getLogger()

    # model
    logger.info('setup and restore model')
    net = model_factory[model_type](NUM_CLASSES)
    net.load_state_dict(torch.load(weight_pth))
    net.cuda()

    is_dist = dist.is_initialized()
    if is_dist:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank, ], output_device=local_rank)

    # evaluator
    heads, mious = eval_model(net=net, ims_per_gpu=ims_per_gpu, im_root=im_root, im_anns=val_im_anns,
                              false_analysis_path=false_analysis_path, crop_size=crop_size)
    logger.info(tabulate([mious], headers=heads, tablefmt='orgtbl'))


if __name__ == "__main__":
    args = parse_args()

    with open(args.config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if not args.local_rank == -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:{}'.format(args.port),
                                world_size=torch.cuda.device_count(),
                                rank=args.local_rank
                                )

    if not osp.exists(args.log_path):
        os.makedirs(args.log_path)

    setup_logger('{}-eval'.format(args.model), args.log_path)

    evaluate(ims_per_gpu=cfg['ims_per_gpu'], crop_size=cfg['crop_size'], weight_pth=args.weight_pth,
             model_type=args.model, im_root=args.im_root, val_im_anns=args.val_im_anns,
             false_analysis_path=args.false_analysis_path)
