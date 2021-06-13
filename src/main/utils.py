import os

import torch
from torch import nn
import torch.distributed as dist

from src.model_components.architectures import model_factory
from src.main.consts import NUM_CLASSES


def get_next_dir_name(root_dir):
    i = 0
    path = os.path.join(root_dir, str(i))
    while os.path.exists(path):
        i += 1
        path = os.path.join(root_dir, str(i))
    os.mkdir(path)

    return path


def get_next_file_name(root_dir, prefix, suffix):
    i = 0
    while os.path.exists(os.path.join(root_dir, f"{prefix}{i}{suffix}")):
        i += 1

    return os.path.join(root_dir, f"{prefix}{i}{suffix}")


def get_model(model_type, num_classes=NUM_CLASSES, is_distributed=True, model_to_load=None, is_train=True,
              use_sync_bn=False):
    net = model_factory[model_type](num_classes)

    if model_to_load is not None:
        net.load_state_dict(torch.load(model_to_load))
    if use_sync_bn:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    net.cuda()  # Moves all model parameters and buffers to the GPU
    if is_train:
        net.train()  # Sets the module in training mode
    else:
        net.eval()

    if is_distributed:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank, ], output_device=local_rank)

    return net
