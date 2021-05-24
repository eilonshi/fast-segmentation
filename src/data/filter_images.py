import os
import cv2
import numpy as np
from pathlib import Path
import random
import shutil

from consts import OTHER_LABEL, BUILDING_LABEL, MAPPING_DICT_ADE, IRRELEVANT_LABELS_ADE, MAPPING_DICT_CITYSCAPES, \
    IRRELEVANT_LABELS_CITYSCAPES, MAPPING_DICT_BARAK, IRRELEVANT_LABELS_BARAK, TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT


def filter_data_folder(mapping_dict_, irrelevant_labels_, old_ann_folder_, new_ann_folder_, old_img_folder_,
                       new_img_folder_):
    image_counter = 0
    relevant_images_counter = 0

    # run on all the annotations in the folder
    list_dir = list(Path(old_ann_folder_).rglob("*.png"))
    for ann_path in list_dir:
        ann_old_path = str(ann_path)
        filename = ann_path.name

        if 'cityscapes' in ann_old_path and 'label' not in ann_old_path:
            continue

        image_counter += 1

        old_ann = np.asarray(cv2.imread(ann_old_path))

        # filter irrelevant images (by irrelevant annotations)
        if len(irrelevant_labels_) is not []:
            if np.in1d(old_ann, np.asarray(irrelevant_labels_)).any():
                continue

        # create correct annotation image
        new_ann = np.zeros_like(old_ann) + OTHER_LABEL

        for new_label, old_labels in mapping_dict_.items():
            for old_label in old_labels:
                new_ann[old_ann == old_label] = new_label

        # save annotation and matching image
        if BUILDING_LABEL in new_ann:
            relevant_images_counter += 1
            print(relevant_images_counter)
            new_path = os.path.join(new_ann_folder_, filename)
            cv2.imwrite(new_path, new_ann)

            # copy the image of the annotation
            img_old_path = ann_old_path.replace(old_ann_folder_, old_img_folder_)
            if 'ADE' in ann_old_path:
                img_old_path = img_old_path.replace('png', 'jpg')
            elif 'cityscapes' in ann_old_path:
                img_old_path = img_old_path.replace('gtFine_labelIds', 'leftImg8bit')
            elif 'barak' in ann_old_path:
                img_old_path = img_old_path.replace('png', 'JPG')
                img_old_path = img_old_path.replace('_w', '')

            img_new_path = os.path.join(new_img_folder_, filename)
            shutil.copy(img_old_path, img_new_path)

    print(f'There are {image_counter} images in the {old_img_folder_} data-set, {relevant_images_counter} of them are '
          f'relevant')


def create_train_val_test_txt_files(output_path, data_directories, train_val_test_ratio):
    train_thresh = train_val_test_ratio[0]
    val_thresh = train_thresh + train_val_test_ratio[1]

    random.seed(123)

    images_directory = 'relevant_images'
    annotations_directory = 'relevant_annotations'

    train_file = os.path.join(output_path, 'train.txt')
    open(train_file, 'w').close()
    val_file = os.path.join(output_path, 'val.txt')
    open(val_file, 'w').close()
    test_file = os.path.join(output_path, 'test.txt')
    open(test_file, 'w').close()

    # run on each data directory
    for directory in data_directories:
        images_path = os.path.join(base_path, directory, images_directory)
        image_filenames = [f for f in os.listdir(images_path)]

        # run on each image and annotation in the directory
        for img_name in image_filenames:
            ann_name = img_name
            if 'ADE' in directory:
                ann_name = ann_name.replace('jpg', 'png')

            prob = random.uniform(0, 1)

            if prob < train_thresh:
                file_to_write = train_file
            elif prob < val_thresh:
                file_to_write = val_file
            else:
                file_to_write = test_file

            img_path = os.path.join(directory, images_directory, img_name)
            ann_path = os.path.join(directory, annotations_directory, ann_name)

            # save the image and annotation in the file
            with open(file_to_write, 'a') as file:
                file.write(img_path + ',' + ann_path + '\n')


if __name__ == '__main__':
    # ADE
    base_path = 'data/ADE20k_outdoors'
    old_img_folder = os.path.join(base_path, 'images/training')
    new_img_folder = os.path.join(base_path, 'relevant_images')
    old_ann_folder = os.path.join(base_path, 'annotations/training')
    new_ann_folder = os.path.join(base_path, 'relevant_annotations')
    mapping_dict = MAPPING_DICT_ADE
    irrelevant_labels = IRRELEVANT_LABELS_ADE
    filter_data_folder(mapping_dict, irrelevant_labels, old_ann_folder, new_ann_folder, old_img_folder, new_img_folder)

    # CITYSCAPES
    base_path = 'data/cityscapes'
    old_img_folder = os.path.join(base_path, 'leftImg8bit')
    new_img_folder = os.path.join(base_path, 'relevant_images')
    old_ann_folder = os.path.join(base_path, 'gtFine')
    new_ann_folder = os.path.join(base_path, 'relevant_annotations')
    mapping_dict = MAPPING_DICT_CITYSCAPES
    irrelevant_labels = IRRELEVANT_LABELS_CITYSCAPES
    filter_data_folder(mapping_dict, irrelevant_labels, old_ann_folder, new_ann_folder, old_img_folder, new_img_folder)

    # BARAK
    base_path = 'data/barak'
    old_img_folder = os.path.join(base_path, 'images')
    new_img_folder = os.path.join(base_path, 'relevant_images')
    old_ann_folder = os.path.join(base_path, 'annotations')
    new_ann_folder = os.path.join(base_path, 'relevant_annotations')
    mapping_dict = MAPPING_DICT_BARAK
    irrelevant_labels = IRRELEVANT_LABELS_BARAK
    filter_data_folder(mapping_dict, irrelevant_labels, old_ann_folder, new_ann_folder, old_img_folder, new_img_folder)

    # create train, val, test txt files
    base_path = 'data'

    data_dirs = [
        'ADE20k_outdoors',
        'cityscapes',
        'barak'
    ]

    create_train_val_test_txt_files(output_path=base_path, data_directories=data_dirs,
                                    train_val_test_ratio=[TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT])
