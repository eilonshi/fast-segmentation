import os

import cv2
import numpy as np

print(f'I am running at {os.getcwd()}')

directory = 'annotations/training'
image_counter = 0
with open('train.txt', 'w') as file:
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_counter += 1
            image_path = os.path.join(directory, filename)

            image = np.asarray(cv2.imread(image_path))
            print(np.unique(image))

        else:
            continue


print(f'There are {image_counter} images in the ADE20K outdoors data-set')
