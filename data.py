import numpy as np
import os
import cv2
import random

from numpy.lib.type_check import imag

"""

Create a dataset containing desired amount of images.
    @param {str} img_folder_path : path of folder containing images (CelebA).
    @param {int} size : numbers of photos in the dataset created.
    @return {List[np.array]} img_data_list: List containing arrays that store images.
    @return {List[str]} img_name: List containing name of each images in the dataset.

"""
def create_dataset(img_folder_path, size : int):
    img_data_list = []
    img_name = []

    for i in range(size):
        file = random.choice(os.listdir(img_folder_path))
        image_path = os.path.join(img_folder_path, file)
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        image = image.astype('float32')
        img_data_list.append(image)
        img_name.append(file)
    
    return img_data_list, img_name

