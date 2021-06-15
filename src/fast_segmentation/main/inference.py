import argparse
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple

import yaml

from src.fast_segmentation.model_components.data_cv2 import TransformationVal
from src.fast_segmentation.model_components.transform_cv2 import ToTensor
from src.fast_segmentation.main.utils import build_model
from src.fast_segmentation.visualization.visualize import save_labels_mask_with_legend

torch.set_grad_enabled(False)


def parse_args():
    """
    Creates the parser for inference arguments

    Returns:
        The parser
    """
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=str, default='bisenetv2')
    parse.add_argument('--weight-path', type=str,
                       default='/home/bina/PycharmProjects/fast-segmentation/models/5/best_model.pth')
    parse.add_argument('--demo-path', type=str,
                       default='/home/bina/PycharmProjects/fast-segmentation/data/inference_results')
    parse.add_argument('--demo_im_anns', type=str,
                       default='/home/bina/PycharmProjects/fast-segmentation/data/demo.txt')
    parse.add_argument('--im_root', type=str, default='/home/bina/PycharmProjects/fast-segmentation/data')
    parse.add_argument('--config_path', type=str,
                       default='/home/bina/PycharmProjects/fast-segmentation/configs/main_cfg.yaml')

    return parse.parse_args()


def read_image_and_label(demo_im_anns: str, im_root: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads image and label according to the first line in the given directory

    Args:
        demo_im_anns: the relative path to the file from the root data directory
        im_root: the root directory for the data

    Returns:
        image (rgb) and corresponding label (grayscale)
    """
    with open(demo_im_anns) as ann_file:
        first_line = ann_file.readline()

    img_and_label = str.split(first_line, ',')

    image_path = os.path.join(im_root, img_and_label[0]).rstrip()
    label_path = os.path.join(im_root, img_and_label[1]).rstrip()

    image = np.asarray(cv2.imread(image_path))
    label = np.asarray(cv2.imread(label_path, 0))

    return image, label


def create_empty_label(image: np.ndarray) -> np.ndarray:
    """
    Creates an empty annotation mask according to the shape of the given image

    Args:
        image: an image with shape WxHxC (C is channels)

    Returns:
        black mask with the shape of (WxH) by the given image
    """
    return np.zeros(image.shape[:2])


def preprocess_image(image: np.ndarray, crop_size: Tuple[int, int]) -> torch.Tensor:
    """
    Converts the given image to a pytorch tensor and makes some operations on it

    Args:
        crop_size:
        image: rgb image with shape WxHxC

    Returns:
        pytorch tensor image with 4 dimensions (defined by the crop size)
    """
    label = create_empty_label(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_label = {'image': image_rgb, 'label': label}
    image_label_cropped = TransformationVal(crop_size=crop_size)(image_label)

    image_tensor = ToTensor()(image_label_cropped)['image']
    image_tensor = torch.unsqueeze(image_tensor, 0)
    image_tensor = image_tensor.cuda()

    # TODO: test that the processed image is the same as in the train

    return image_tensor


def inference(image: np.ndarray, model_type: str, weight_path: str, demo_path: str, crop_size: Tuple[int, int],
              label: np.ndarray = None):
    """
    The main function that responsible of applying the semantic segmentation model on the given image, the result is
    saved to the corresponding paths

    Args:
        crop_size:
        weight_path:
        demo_path:
        image: an image to run the segmentation model on
        model_type: the name of the model architecture type
        label: optional - an annotation mask to save next to the result

    Returns:
        None
    """
    net = build_model(model_type=model_type, is_distributed=False, pretrained_model_path=weight_path,
                      is_train=False, use_sync_bn=False)

    image_tensor = preprocess_image(image, crop_size=crop_size)

    # get output from logits
    logits, *logits_aux = net(image_tensor)
    out = logits[:1].argmax(dim=1).squeeze().detach().cpu().numpy()

    # save image, label and inference
    plt.imsave(os.path.join(demo_path, 'inf_image.jpg'), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    save_labels_mask_with_legend(mask=out, save_path=os.path.join(demo_path, 'inf_result.jpg'))
    if label is not None:
        save_labels_mask_with_legend(mask=label, save_path=os.path.join(demo_path, 'inf_label.jpg'))


if __name__ == '__main__':
    args = parse_args()

    with open(args.config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    image_original, label_original = read_image_and_label(demo_im_anns=args.demo_im_anns, im_root=args.im_root)
    inference(image=image_original, label=label_original, model_type=args.model, weight_path=args.weight_path,
              demo_path=args.demo_path, crop_size=cfg['crop_size'])
