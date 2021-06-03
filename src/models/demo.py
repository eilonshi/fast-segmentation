import argparse
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.lib.architectures import model_factory
from src.lib.tevel_cv2 import get_data_loader
from src.lib import transform_cv2 as t
from src.configs import cfg_factory
from src.models.consts import NUM_CLASSES
from src.visualization.visualize import save_labels_mask_with_legend

torch.set_grad_enabled(False)
np.random.seed(123)


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2')
    parse.add_argument('--weight-path', type=str,
                       default='/home/bina/PycharmProjects/tevel-segmentation/models/model_final_0.pth')
    parse.add_argument('--demo-path', dest='demo_path', type=str,
                       default='/home/bina/PycharmProjects/tevel-segmentation/data/demo_results')

    return parse.parse_args()


args = parse_args()
cfg = cfg_factory[args.model]


def load_model():
    net = model_factory[cfg.model_type](NUM_CLASSES)
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    net.eval()
    net.cuda()

    return net


def get_image_and_label():
    with open(cfg.demo_im_anns) as ann_file:
        first_line = ann_file.readline()
    img_and_label = str.split(first_line, ',')
    image_original_path = os.path.join(cfg.im_root, img_and_label[0]).rstrip()
    label_original_path = os.path.join(cfg.im_root, img_and_label[1]).rstrip()
    image_original = np.asarray(cv2.imread(image_original_path))
    label_original = np.asarray(cv2.imread(label_original_path, 0))

    return image_original, label_original


def inference():
    net = load_model()
    data_loader = get_data_loader(cfg.im_root, cfg.demo_im_anns, cfg.ims_per_gpu, cfg.scales, cfg.crop_size,
                                  cfg.max_iter, mode='val', distributed=False)

    image_original, label_original = get_image_and_label()

    im_lb = dict(im=image_original, lb=label_original)
    trans_func = t.RandomResizedCrop(size=cfg.crop_size, is_random=False)
    im_lb = trans_func(im_lb)

    for iteration, (image, label) in enumerate(data_loader):
        image = image.cuda()

        # get logits
        logits, *logits_aux = net(image)
        out = logits[:1].argmax(dim=1).squeeze().detach().cpu().numpy()

        # save image, label and inference
        plt.imsave(os.path.join(args.demo_path, 'image.jpg'), cv2.cvtColor(im_lb['im'], cv2.COLOR_BGR2RGB))
        save_labels_mask_with_legend(mask=out, save_path=os.path.join(args.demo_path, 'result.jpg'))
        save_labels_mask_with_legend(mask=im_lb['lb'], save_path=os.path.join(args.demo_path, 'label.jpg'))


if __name__ == '__main__':
    inference()
