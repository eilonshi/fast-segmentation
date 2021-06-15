import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple

from src.fast_segmentation.main.consts import LABEL_TO_COLOR, IGNORE_LABEL, OTHER_LABEL, PIXELS_SCALE


def labels_mask_to_colored_image(mask: np.ndarray) -> np.ndarray:
    """
    Converts the mask to an image - replaces the labels with the matching colors

    Args:
        mask: a grayscale semantic segmentation annotation mask with values up to the number of classes
              (like in the consts)

    Returns:
        colored annotation mask - rgb
    """
    return np.asarray(list(LABEL_TO_COLOR.values()), dtype=np.uint8)[mask]


def get_legends(colors: List[Tuple[int]]) -> List[Rectangle]:
    """
    Creates colored rects for legends of image

    Args:
        colors: a list of colors - a color is a tuple of rgb

    Returns:
        a list of colored rectangles
    """
    legends = [Rectangle((0, 0), 1, 1, color=np.asarray(color[::-1]) / PIXELS_SCALE) for color in colors]

    return legends


def save_image_with_legends_and_labels(save_path: str, image: np.ndarray, legends: List[Rectangle], labels: List[str]):
    """
    Saves the given image with the given legends rects and labels

    Args:
        save_path: the path to the folder to save the image
        image: the image to put the legends on
        legends: a list of colored rects
        labels: a list of the names of the legends

    Returns:
        None
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.legend(legends, labels)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def save_labels_mask_with_legend(mask: np.ndarray, save_path: str):
    """
    Converts the mask to a colored rgb mask and saves the new mask to the given path

    Args:
        mask: a grayscale mask with values up to the number of classes (like in the consts)
        save_path: the path to save the new mask

    Returns:
        None
    """
    mask[mask == IGNORE_LABEL] = OTHER_LABEL
    image = labels_mask_to_colored_image(mask)
    labels = list(LABEL_TO_COLOR.keys())
    legends = get_legends(list(LABEL_TO_COLOR.values()))

    save_image_with_legends_and_labels(save_path, image, legends, labels)
