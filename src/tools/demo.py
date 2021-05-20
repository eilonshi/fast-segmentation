import sys

sys.path.insert(0, '')

import argparse
import torch
import numpy as np
import cv2

from src import lib as T
from src.lib.models import model_factory
from src.configs import cfg_factory
import time

torch.set_grad_enabled(False)
np.random.seed(123)

# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2', )
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth', )
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png', )
parse.add_argument('--res-path', dest='res_path', type=str, default='./res.jpg', )
args = parse.parse_args()
cfg = cfg_factory[args.model]

palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](19)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223),  # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
im = cv2.imread(args.img_path)[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

# inference
time0 = time.time()
out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
time1 = time.time()
print(f'Inference time on GTX 1080 Ti:    {time1 - time0:.2f} seconds')

# save inference
pred = palette[out]
cv2.imwrite(args.res_path, pred)
