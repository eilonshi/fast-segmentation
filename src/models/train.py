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
from src.lib.soft_dice_loss import SoftDiceLoss
from src.models.consts import NUM_CLASSES
from src.lib.architectures import model_factory
from src.lib.tevel_cv2 import get_data_loader
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
    parse.add_argument('--finetune-from', type=str, default='../../models/3/best_model.pth')
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
    criteria_pre = SoftDiceLoss()
    criteria_aux = [SoftDiceLoss() for _ in range(cfg.num_aux_heads)]

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
    optimizer = torch.optim.Adam(
        params_list,
        lr=cfg.lr_start,
        betas=cfg.optimizer_betas,
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


def log_ious(writer, mious, iteration, heads, logger, mode):
    single_scale_miou, single_scale_crop_miou, ms_flip_miou, ms_flip_crop_miou = mious
    writer.add_scalar(f"mIOU/{mode}/single_scale", single_scale_miou, iteration)
    writer.add_scalar(f"mIOU/{mode}/single_scale_crop", single_scale_crop_miou, iteration)
    writer.add_scalar(f"mIOU/{mode}/multi_scale_flip", ms_flip_miou, iteration)
    writer.add_scalar(f"mIOU/{mode}/multi_scale_flip_crop", ms_flip_crop_miou, iteration)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))


def get_next_dir_name(root_dir):
    i = 0
    path = os.path.join(root_dir, str(i))
    while os.path.exists(path):
        i += 1
        path = os.path.join(root_dir, str(i))
    os.mkdir(path)

    return path


def train():
    logger = logging.getLogger()
    tensorboard_log_dir = get_next_dir_name(
        root_dir=cfg.tensorboard_path)
    models_dir = get_next_dir_name(
        root_dir=cfg.models_path)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    is_dist = dist.is_initialized()

    # set all components
    data_loader = get_data_loader(cfg.im_root, cfg.train_im_anns, cfg.ims_per_gpu, cfg.scales, cfg.crop_size,
                                  cfg.max_iter, mode='train', distributed=is_dist)
    net, criteria_pre, criteria_aux = set_model()
    optimizer = set_optimizer(net)
    scaler = amp.GradScaler()  # mixed precision training
    net = set_model_dist(net)  # distributed training
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()  # metrics
    lr_scheduler = WarmupPolyLrScheduler(optimizer, power=0.9, max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
                                         warmup_ratio=0.1, warmup='exp', last_epoch=-1, )
    best_score = 0

    # train loop
    for iteration, (image, label) in enumerate(data_loader):
        image = image.cuda()
        label = label.cuda()

        if iteration == 0:
            writer.add_graph(net, image)

        label = torch.squeeze(label, 1)

        optimizer.zero_grad()
        with amp.autocast(enabled=cfg.use_fp16):
            # get main loss and auxiliary losses
            logits, *logits_aux = net(image)
            loss_pre = criteria_pre(logits, label)
            loss_aux = [criteria(logits, label) for criteria, logits in zip(criteria_aux, logits_aux)]

            loss = loss_pre + sum(loss_aux)
            writer.add_scalar("Loss/train", loss, iteration)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
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
            while os.path.exists(os.path.join(models_dir, f"model_final_{i}.pth")):
                i += 1
            log_pth = os.path.join(cfg.log_path, f"model_final_{i}.pth")
            save_pth = os.path.join(models_dir, f"model_final_{i}.pth")
            logger.info('\nsave models to {}'.format(log_pth))
            state = net.module.state_dict()
            if dist.get_rank() == 0:
                torch.save(state, save_pth)

            # evaluate the results
            logger.info('\nevaluating the model')
            torch.cuda.empty_cache()
            heads_val, mious_val = eval_model(net=net, ims_per_gpu=cfg.ims_per_gpu, im_root=cfg.im_root,
                                              im_anns=cfg.val_im_anns)
            log_ious(writer, mious_val, iteration, heads_val, logger, mode='val')
            heads_train, mious_train = eval_model(net=net, ims_per_gpu=cfg.ims_per_gpu, im_root=cfg.im_root,
                                                  im_anns=cfg.train_im_anns)
            log_ious(writer, mious_train, iteration, heads_train, logger, mode='train')

            if mious_val[0] > best_score:
                best_score = mious_val[0]
                save_pth = os.path.join(models_dir, f"best_model.pth")
                state = net.module.state_dict()
                if dist.get_rank() == 0:
                    torch.save(state, save_pth)

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
    if not osp.exists(cfg.log_path):
        os.makedirs(cfg.log_path)
    setup_logger('{}-train'.format(cfg.model_type), cfg.log_path)
    train()


if __name__ == "__main__":
    main()
