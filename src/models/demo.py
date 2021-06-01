import argparse
import os

import torch
import numpy as np
import cv2
import time

from src.lib import transform_cv2 as T
from src.lib.architectures import model_factory
from src.configs import cfg_factory
from src.lib.cityscapes_cv2 import get_data_loader
from src.models.consts import NUM_CLASSES

torch.set_grad_enabled(False)
np.random.seed(123)

# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2')
parse.add_argument('--weight-path', type=str,
                   default='/home/bina/PycharmProjects/tevel-segmentation/models/model_final_211.pth')
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

data_loader = get_data_loader(cfg.im_root, cfg.demo_im_anns, cfg.ims_per_gpu, cfg.scales, cfg.crop_size,
                              cfg.max_iter, mode='val', distributed=False)
img_path = os.path.join(os.path.dirname(args.res_path), 'img.jpg')
label_path = os.path.join(os.path.dirname(args.res_path), 'label.jpg')

for iteration, (image, label) in enumerate(data_loader):
    image = image.cuda()
    label = label.cuda()

    label = torch.squeeze(label, 1)

    # get logits
    time0 = time.time()
    logits, *logits_aux = net(image)
    time1 = time.time()
    print(f'Inference time on GTX 1080 Ti:    {time1 - time0:.2f} seconds')
    out = logits[:1].argmax(dim=1).squeeze().detach().cpu().numpy()
    label = label[:1].squeeze().detach().cpu().numpy()

    # save inference
    pred = palette[out]
    cv2.imwrite(args.res_path, pred)
    img = cv2.imread(args.img_path)
    cv2.imwrite(img_path, img)
    label_to_show = palette[label]
    cv2.imwrite(label_path, label_to_show)
