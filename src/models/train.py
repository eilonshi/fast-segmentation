import os
import os.path as osp
import random
import logging
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

from evaluate import eval_model
from src.configs import cfg_factory
from src.models.utils import get_next_dir_name, get_next_file_name, get_model
from src.lib.soft_dice_loss import SoftDiceLoss
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


def get_losses():
    criteria_pre = SoftDiceLoss()
    criteria_aux = [SoftDiceLoss() for _ in range(cfg.num_aux_heads)]

    return criteria_pre, criteria_aux


def set_optimizer(model):
    wd_params, non_wd_params = [], []

    for name, param in model.named_parameters():
        if param.dim() == 1:
            non_wd_params.append(param)
        elif param.dim() == 2 or param.dim() == 4:
            wd_params.append(param)

    params_list = [
        {'params': wd_params},
        {'params': non_wd_params, 'weight_decay': 0}
    ]
    optimizer = torch.optim.Adam(params_list, lr=cfg.lr_start, betas=cfg.optimizer_betas, weight_decay=cfg.weight_decay)

    return optimizer


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i)) for i in range(cfg.num_aux_heads)]

    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def log_ious(writer, mious, iteration, heads, logger, mode):
    single_scale_miou, single_scale_crop_miou, ms_flip_miou, ms_flip_crop_miou = mious
    writer.add_scalar(f"mIOU/{mode}/single_scale", single_scale_miou, iteration)
    writer.add_scalar(f"mIOU/{mode}/single_scale_crop", single_scale_crop_miou, iteration)
    writer.add_scalar(f"mIOU/{mode}/multi_scale_flip", ms_flip_miou, iteration)
    writer.add_scalar(f"mIOU/{mode}/multi_scale_flip_crop", ms_flip_crop_miou, iteration)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))


def save_best_model(mious_val, best_score, models_dir, net):
    if mious_val[0] > best_score:
        best_score = mious_val[0]
        save_pth = os.path.join(models_dir, f"best_model.pth")
        state = net.module.state_dict()
        if dist.get_rank() == 0:
            torch.save(state, save_pth)

    return best_score


def save_checkpoint(models_dir, logger, net, writer, iteration, best_score):
    log_pth = get_next_file_name(cfg.log_path, prefix='model_final_', suffix='.pth')
    save_pth = get_next_file_name(models_dir, prefix='model_final_', suffix='.pth')

    logger.info('\nsave models to {}'.format(log_pth))
    state = net.module.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)

    logger.info('\nevaluating the model')
    torch.cuda.empty_cache()

    # evaluate val set
    heads_val, mious_val = eval_model(net=net, ims_per_gpu=cfg.ims_per_gpu, im_root=cfg.im_root,
                                      im_anns=cfg.val_im_anns)
    log_ious(writer, mious_val, iteration, heads_val, logger, mode='val')

    # evaluate train set
    heads_train, mious_train = eval_model(net=net, ims_per_gpu=cfg.ims_per_gpu, im_root=cfg.im_root,
                                          im_anns=cfg.train_im_anns)
    log_ious(writer, mious_train, iteration, heads_train, logger, mode='train')

    # save best model
    best_score = save_best_model(mious_val, best_score, models_dir, net)

    return best_score


def train():
    logger = logging.getLogger()
    tensorboard_log_dir = get_next_dir_name(root_dir=cfg.tensorboard_path)
    models_dir = get_next_dir_name(root_dir=cfg.models_path)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    is_dist = dist.is_initialized()

    # set all components
    data_loader = get_data_loader(cfg.im_root, cfg.train_im_anns, cfg.ims_per_gpu, cfg.scales, cfg.crop_size,
                                  cfg.max_iter, mode='train', distributed=is_dist)
    net = get_model(cfg.model_type, is_train=True, is_distributed=is_dist, model_to_load=args.finetune_from,
                    use_sync_bn=cfg.use_sync_bn)
    criteria_pre, criteria_aux = get_losses()
    optimizer = set_optimizer(net)
    scaler = amp.GradScaler()  # mixed precision training
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

        with amp.autocast(enabled=cfg.use_fp16):  # get main loss and auxiliary losses
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
        _ = [metric.update(loss.item()) for metric, loss in zip(loss_aux_meters, loss_aux)]

        # print training log message
        if (iteration + 1) % cfg.message_iters == 0:
            lr = lr_scheduler.get_lr()
            lr = sum(lr) / len(lr)
            print_log_msg(iteration, cfg.max_iter, lr, time_meter, loss_meter, loss_pre_meter, loss_aux_meters)

        # saving the model and evaluating it
        if (iteration + 1) % cfg.checkpoint_iters == 0:
            best_score = save_checkpoint(models_dir, logger, net, writer, iteration, best_score)

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
