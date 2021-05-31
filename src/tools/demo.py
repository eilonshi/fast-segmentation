import math
import sys

sys.path.insert(0, '')

import argparse
import torch
import numpy as np
import cv2
import time

from src.lib import transform_cv2 as T
from src.lib.models import model_factory
from src.configs import cfg_factory
from src.tools.consts import NUM_CLASSES

torch.set_grad_enabled(False)
np.random.seed(123)

# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2')
parse.add_argument('--weight-path', type=str,
                   default='/home/bina/PycharmProjects/tevel-segmentation/models/model_final_146.pth')
parse.add_argument('--img-path', dest='img_path', type=str,
                   default='/home/bina/PycharmProjects/tevel-segmentation/data/ADE20k_outdoors/relevant_images/ADE_train_00014944.png')
parse.add_argument('--ann-path', dest='ann_path', type=str,
                   default='/home/bina/PycharmProjects/tevel-segmentation/data/ADE20k_outdoors/relevant_annotations/ADE_train_00014944.png')
parse.add_argument('--res-path', dest='res_path', type=str,
                   default='/home/bina/PycharmProjects/tevel-segmentation/data/data_examples/res.jpg')
args = parse.parse_args()
cfg = cfg_factory[args.model]

palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](NUM_CLASSES)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223),  # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)

im = cv2.imread(args.img_path)[:, :, ::-1]
crop_h, crop_w = [512, 1024]
scales = [0.25, 2]
scale = np.random.uniform(min(scales), max(scales))
im_h, im_w = [math.ceil(el * scale) for el in im.shape[:2]]
im = cv2.resize(im, (im_w, im_h))

pad_h, pad_w = 0, 0
if im_h < crop_h:
    pad_h = (crop_h - im_h) // 2 + 1
if im_w < crop_w:
    pad_w = (crop_w - im_w) // 2 + 1
if pad_h > 0 or pad_w > 0:
    im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))

im_h, im_w, _ = im.shape
sh, sw = np.random.random(2)
sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))

im = im[sh:sh + crop_h, sw:sw + crop_w, :].copy()

im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

# inference
time0 = time.time()
out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
time1 = time.time()
print(f'Inference time on GTX 1080 Ti:    {time1 - time0:.2f} seconds')

# save inference
pred = palette[out]
cv2.imwrite(args.res_path, pred)
