import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from src.models.consts import LABEL_TO_COLOR, IGNORE_LABEL, OTHER_LABEL


def save_labels_mask_with_legend(mask, save_path):
    mask[mask == IGNORE_LABEL] = OTHER_LABEL
    image = np.asarray(list(LABEL_TO_COLOR.values()), dtype=np.uint8)[mask]
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    handles = [Rectangle((0, 0), 1, 1, color=np.asarray(color[::-1]) / 255) for color in LABEL_TO_COLOR.values()]
    labels = [label for label in LABEL_TO_COLOR.keys()]
    plt.legend(handles, labels)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
