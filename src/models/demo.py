import argparse
import os

import torch
import numpy as np
import cv2
import time

from src.lib.architectures import model_factory
from src.configs import cfg_factory
from src.lib.cityscapes_cv2 import get_data_loader
from src.models.consts import NUM_CLASSES, LABEL_TO_COLOR

torch.set_grad_enabled(False)
np.random.seed(123)

# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2')
parse.add_argument('--weight-path', type=str,
                   default='/home/bina/PycharmProjects/tevel-segmentation/models/model_final_0.pth')
parse.add_argument('--img-path', dest='img_path', type=str,
                   default='/home/bina/PycharmProjects/tevel-segmentation/data/demo_results/img.jpg')
parse.add_argument('--label-path', dest='label_path', type=str,
                   default='/home/bina/PycharmProjects/tevel-segmentation/data/demo_results/label.jpg')
parse.add_argument('--res-path', dest='res_path', type=str,
                   default='/home/bina/PycharmProjects/tevel-segmentation/data/demo_results/res.jpg')
parse.add_argument('--legend-path', dest='legend_path', type=str,
                   default='/home/bina/PycharmProjects/tevel-segmentation/data/demo_results/legend.jpg')
args = parse.parse_args()
cfg = cfg_factory[args.model]

# define model
net = model_factory[cfg.model_type](NUM_CLASSES)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

# define data loader
data_loader = get_data_loader(cfg.im_root, cfg.demo_im_anns, cfg.ims_per_gpu, cfg.scales, cfg.crop_size,
                              cfg.max_iter, mode='val', distributed=False)

# load image and label
img_path = args.img_path
with open(cfg.demo_im_anns) as ann_file:
    first_line = ann_file.readline()
img_and_label = str.split(first_line, ',')
image_original_path = os.path.join(cfg.im_root, img_and_label[0]).rstrip()
label_original_path = os.path.join(cfg.im_root, img_and_label[1]).rstrip()
image_original = np.asarray(cv2.resize(cv2.imread(image_original_path), cfg.crop_size[::-1]))
label_original = np.asarray(cv2.resize(cv2.imread(label_original_path), cfg.crop_size[::-1]))

label_path = args.label_path

for iteration, (image, label) in enumerate(data_loader):
    image = image.cuda()
    label = label.cuda()

    label = torch.squeeze(label, 1)

    # get logits
    time0 = time.time()
    logits, *logits_aux = net(image)
    time1 = time.time()
    print(f'Inference time:    {time1 - time0:.2f} seconds')
    out = logits[:1].argmax(dim=1).squeeze().detach().cpu().numpy()
    label = label[:1].squeeze().detach().cpu().numpy()

    # save image, label and inference
    cv2.imwrite(img_path, image_original)
    pred = np.asarray(LABEL_TO_COLOR, dtype=np.uint8)[out]
    cv2.imwrite(args.res_path, pred)
    label_to_show = np.asarray(LABEL_TO_COLOR, dtype=np.uint8)[label_original]
    cv2.imwrite(label_path, label_to_show)

    # save a legend of the colors
    legend = np.zeros_like(image_original)
    legend_shape = legend.shape
    for i, color in enumerate(LABEL_TO_COLOR):
        legend[:, legend_shape[1] // NUM_CLASSES * i: legend_shape[1] // NUM_CLASSES * (i + 1) - 1, :] = color
    cv2.imwrite(args.legend_path, legend)
