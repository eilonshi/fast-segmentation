import os

import cv2
import numpy as np
import shutil

from consts import OTHER_LABEL, BUILDING_LABEL

IRRELEVANT_LABELS_ADE = [27, 62, 91, 114, 129]

mapping_dict = {
    1: [10, 14, 17, 30, 35, 47, 69, 95],
    2: [5, 18, 67, 73],
    3: [1, 2, 4, 6, 26, 49, 85],
    4: [87, 89, 115],
    5: [33],
    6: [3],
    7: [7, 55],
    8: [12, 53, 92],
    9: [9, 15, 19],
    10: [21, 84, 104, 117, 128, 134],
    11: [13],
}

old_img_folder = 'images/training'
new_img_folder = 'relevant_images/training'
old_ann_folder = 'annotations/training'
new_ann_folder = 'relevant_annotations/training'

image_counter = 0
relevant_images_counter = 0

for filename in os.listdir(old_ann_folder):
    if filename.endswith(".png"):
        image_counter += 1
        ann_path = os.path.join(old_ann_folder, filename)

        old_ann = np.asarray(cv2.imread(ann_path))

        if np.in1d(old_ann, np.asarray(IRRELEVANT_LABELS_ADE)).any():
            continue

        new_ann = np.zeros_like(old_ann) + OTHER_LABEL

        for new_label, old_labels in mapping_dict.items():
            for old_label in old_labels:
                new_ann[old_ann == old_label] = new_label

        if BUILDING_LABEL in new_ann:
            relevant_images_counter += 1
            print(relevant_images_counter)
            new_path = os.path.join(new_ann_folder, filename)
            cv2.imwrite(new_path, new_ann)

            # copy the image of the annotation
            img_filename = filename.replace('png', 'jpg')
            img_old_path = os.path.join(old_img_folder, img_filename)
            shutil.copy(img_old_path, new_img_folder)

    else:
        continue

print(f'There are {image_counter} images in the ADE20K outdoors data-set, {relevant_images_counter} of them are '
      f'relevant')
