import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from src.main.consts import LABEL_TO_COLOR, IGNORE_LABEL, OTHER_LABEL


def labels_mask_to_colored_image(mask):
    return np.asarray(list(LABEL_TO_COLOR.values()), dtype=np.uint8)[mask]


def get_legends(colors):
    legends = [Rectangle((0, 0), 1, 1, color=np.asarray(color[::-1]) / 255) for color in colors]

    return legends


def save_image_with_legends_and_labels(save_path, image, legends, labels):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.legend(legends, labels)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def save_labels_mask_with_legend(mask, save_path):
    mask[mask == IGNORE_LABEL] = OTHER_LABEL
    image = labels_mask_to_colored_image(mask)
    labels = list(LABEL_TO_COLOR.keys())
    legends = get_legends(list(LABEL_TO_COLOR.values()))

    save_image_with_legends_and_labels(save_path, image, legends, labels)
