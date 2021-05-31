import os
import os.path as osp
import random
import logging
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

from evaluate import eval_model
from src.configs import cfg_factory
from src.models.consts import NUM_CLASSES
from src.lib.architectures import model_factory
from src.lib.cityscapes_cv2 import get_data_loader
from src.lib.ohem_ce_loss import OHEMCrossEntropyLoss
from src.lib.lr_scheduler import WarmupPolyLrScheduler
from src.lib.meters import TimeMeter, AvgMeter
from src.lib.logger import setup_logger, print_log_msg

# fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=0)
    parse.add_argument('--port', dest='port', type=int, default=44554)
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2')
    parse.add_argument('--finetune-from', type=str, default=None)
    parse.add_argument('--amp', type=bool, default=True)
    return parse.parse_args()


args = parse_args()
cfg = cfg_factory[args.model]


def set_model():
    net = model_factory[cfg.model_type](NUM_CLASSES)

    if args.finetune_from is not None:
        net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    if cfg.use_sync_bn:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    net.cuda()
    net.train()
    criteria_pre = OHEMCrossEntropyLoss(0.7)
    criteria_aux = [OHEMCrossEntropyLoss(0.7) for _ in range(cfg.num_aux_heads)]

    return net, criteria_pre, criteria_aux


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optimizer = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    return optimizer


def set_model_dist(net):
    local_rank = dist.get_rank()
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank, ],
        # find_unused_parameters=True,
        output_device=local_rank)
    return net


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
                       for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def train():
    logger = logging.getLogger()
    writer = SummaryWriter(log_dir='/home/bina/PycharmProjects/tevel-segmentation/logs/tensorboard_logs')
    is_dist = dist.is_initialized()

    # dataset
    data_loader = get_data_loader(cfg.im_root, cfg.train_im_anns, cfg.ims_per_gpu, cfg.scales, cfg.crop_size,
                                  cfg.max_iter, mode='train', distributed=is_dist)

    # model
    net, criteria_pre, criteria_aux = set_model()

    # optimizer
    optim = set_optimizer(net)

    # mixed precision training
    scaler = amp.GradScaler()

    # ddp training
    net = set_model_dist(net)

    # meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    # lr scheduler
    lr_scheduler = WarmupPolyLrScheduler(optim, power=0.9, max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
                                         warmup_ratio=0.1, warmup='exp', last_epoch=-1, )

    # train loop
    for iteration, (image, label) in enumerate(data_loader):
        image = image.cuda()
        label = label.cuda()

        label = torch.squeeze(label, 1)

        optim.zero_grad()
        with amp.autocast(enabled=cfg.use_fp16):
            # get main loss and auxiliary losses
            logits, *logits_aux = net(image)
            loss_pre = criteria_pre(logits, label)
            loss_aux = [criteria(logits, label) for criteria, logits in zip(criteria_aux, logits_aux)]

            loss = loss_pre + sum(loss_aux)
            writer.add_scalar("Loss/train", loss, iteration)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        torch.cuda.synchronize()

        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

        # print training log message
        if (iteration + 1) % cfg.message_iters == 0:
            lr = lr_scheduler.get_lr()
            lr = sum(lr) / len(lr)
            print_log_msg(iteration, cfg.max_iter, lr, time_meter, loss_meter, loss_pre_meter, loss_aux_meters)

        # saving the model and evaluating it
        if (iteration + 1) % cfg.checkpoint_iters == 0:
            # save the model
            i = 0
            while os.path.exists(os.path.join(cfg.respth, f"model_final_{i}.pth")):
                i += 1
            log_pth = os.path.join(cfg.logpth, f"model_final_{i}.pth")
            save_pth = os.path.join(cfg.respth, f"model_final_{i}.pth")
            logger.info('\nsave models to {}'.format(log_pth))
            state = net.module.state_dict()
            if dist.get_rank() == 0:
                torch.save(state, save_pth)

            # evaluate the results
            logger.info('\nevaluating the model')
            torch.cuda.empty_cache()
            heads, mious = eval_model(net=net, ims_per_gpu=cfg.ims_per_gpu, im_root=cfg.im_root,
                                      im_anns=cfg.val_im_anns)
            logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))

        lr_scheduler.step()

    writer.flush()
    writer.close()


def main():
    torch.cuda.empty_cache()
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(args.port),
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    if not osp.exists(cfg.respth):
        os.makedirs(cfg.respth)
    setup_logger('{}-train'.format(cfg.model_type), cfg.respth)
    train()


if __name__ == "__main__":
    main()
