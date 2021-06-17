import os
import shutil

import torch
import torch.distributed as dist
from torch import nn

from fast_segmentation.model_components.architectures import model_factory
from fast_segmentation.core.consts import NUM_CLASSES


def get_next_dir_name(root_dir: str) -> str:
    """
    Finds the next index for a new directory name in the given path (searches for integer names for the new directory)
    Args:
        root_dir: the path of the directory to locate the new directory in

    Returns:
        the path for the new directory
    """
    i = 0
    path = os.path.join(root_dir, str(i))

    while os.path.exists(path):
        i += 1
        path = os.path.join(root_dir, str(i))

    os.mkdir(path)

    return path


def get_next_file_name(root_dir: str, prefix: str, suffix: str) -> str:
    """
    Finds a name for a new file in the given folder - supposes that the file has the given prefix and suffix
    Args:
        root_dir: the path of the directory of the new file
        prefix: a string that is at the beginning of the new file name
        suffix: a string that is at the end of the new file name

    Returns:
        the path for the new file
    """
    i = 0

    while os.path.exists(os.path.join(root_dir, f"{prefix}{i}{suffix}")):
        i += 1

    return os.path.join(root_dir, f"{prefix}{i}{suffix}")


def build_model(model_type: str, num_classes: int = NUM_CLASSES, is_distributed: bool = True,
                pretrained_model_path: str = None, is_train: bool = True, use_sync_bn=False) -> nn.Module:
    """
    Builds a model from the given type, if a path to model weights is given then loads the pretrained model
    
    Args:
        model_type: the name of the model architecture
        num_classes: the number of output channels of the neural network 
        is_distributed: a flag for running the model distributed on multiple machines 
        pretrained_model_path: a path for a pretrained model from the same type
        is_train: a flag for the mode of the model (training mode / evaluation mode)
        use_sync_bn: a flag for using SyncBatchNorm

    Returns:
        a pytorch neural network model
    """
    net = model_factory[model_type](num_classes)

    if pretrained_model_path is not None:
        net.load_state_dict(torch.load(pretrained_model_path))
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


def delete_directory_content(dir_path: str):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
