import argparse
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.lib.tevel_cv2 import TransformationVal
from src.configs import cfg_factory
from src.lib.transform_cv2 import ToTensor
from src.models.utils import get_model
from src.visualization.visualize import save_labels_mask_with_legend

torch.set_grad_enabled(False)


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2')
    parse.add_argument('--weight-path', type=str,
                       default='/home/bina/PycharmProjects/tevel-segmentation/models/5/best_model.pth')
    parse.add_argument('--demo-path', dest='demo_path', type=str,
                       default='/home/bina/PycharmProjects/tevel-segmentation/data/demo_results')

    return parse.parse_args()


args = parse_args()
cfg = cfg_factory[args.model]


def read_image_and_label():
    with open(cfg.demo_im_anns) as ann_file:
        first_line = ann_file.readline()
    img_and_label = str.split(first_line, ',')

    image_path = os.path.join(cfg.im_root, img_and_label[0]).rstrip()
    label_path = os.path.join(cfg.im_root, img_and_label[1]).rstrip()

    image = np.asarray(cv2.imread(image_path))
    label = np.asarray(cv2.imread(label_path, 0))

    return image, label


def create_empty_label(image):
    return np.zeros(image.shape[:2])


def preprocess_image(image):
    label = create_empty_label(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_label = {'image': image_rgb, 'label': label}
    image_label_cropped = TransformationVal(cfg.crop_size)(image_label)

    image_tensor = ToTensor()(image_label_cropped)['image']
    image_tensor = torch.unsqueeze(image_tensor, 0)
    image_tensor = image_tensor.cuda()

    # TODO: test that the processed image is the same as in the train

    return image_tensor


def inference(image, label=None):
    net = get_model(model_type=cfg.model_type, is_distributed=False, model_to_load=args.weight_path, is_train=False,
                    use_sync_bn=False)

    image_tensor = preprocess_image(image)

    # get output from logits
    logits, *logits_aux = net(image_tensor)
    out = logits[:1].argmax(dim=1).squeeze().detach().cpu().numpy()

    # save image, label and inference
    plt.imsave(os.path.join(args.demo_path, 'inf_image.jpg'), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    save_labels_mask_with_legend(mask=out, save_path=os.path.join(args.demo_path, 'inf_result.jpg'))
    if label is not None:
        save_labels_mask_with_legend(mask=label, save_path=os.path.join(args.demo_path, 'inf_label.jpg'))


if __name__ == '__main__':
    image_original, label_original = read_image_and_label()
    inference(image_original, label_original)
