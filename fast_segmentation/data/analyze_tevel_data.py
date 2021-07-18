import os.path
from os import listdir

import cv2
import numpy as np

COLOR_RANGE = 256
path = '/home/bina/PycharmProjects/fast-segmentation/data/untagged_tevel_seg_data/VIS'

mean_blue = 0
mean_green = 0
mean_red = 0
std_blue = 0
std_green = 0
std_red = 0

num_images = 0

for image_name in listdir(path):
    num_images += 1
    image_path = os.path.join(path, image_name)
    img = cv2.imread(image_path)

    img_blue = img[0]
    img_green = img[1]
    img_red = img[2]

    mean_blue += np.mean(img_blue)
    mean_green += np.mean(img_green)
    mean_red += np.mean(img_red)

    std_blue += np.std(img_blue)
    std_green += np.std(img_green)
    std_red += np.std(img_red)

print('BGR means: ', [mean_blue / num_images / COLOR_RANGE, mean_green / num_images / COLOR_RANGE,
                      mean_red / num_images / COLOR_RANGE])
print('BGR stds: ',
      [std_blue / num_images / COLOR_RANGE, std_green / num_images / COLOR_RANGE, std_red / num_images / COLOR_RANGE])
