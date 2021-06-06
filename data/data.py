import numpy as np
import os
import cv2
import random
from math import floor
import tensorflow as tf

"""

Create a dataset containing desired amount of images, which splits into train, test and validation dataset.
    @param {str} img_folder_path : path of folder containing images (CelebA).
    @param {int} size : the total numbers of images required to create train, test and validation dataset.
    @param {float} train_split : train dataset size
    @param {float} test_split : test dataset size
    @param {float} valid_split : validation dataset size
    @return {List[np.array]} train_data_list: List containing arrays that store the train dataset.
    @return {List[np.array]} test_data_list: List containing arrays that store the test dataset.
    @return {List[np.array]} valid_data_list: List containing arrays that store the validation dataset.
    @return {List[str]} train_img_name: List containing name of each images in the train dataset.
    @return {List[str]} test_img_name: List containing name of each images in the test dataset.
    @return {List[str]} valid_img_name: List containing name of each images in the validtaion dataset.

"""
def create_split(img_folder_path, train_split, test_split, valid_split, size : int):
    train = floor(train_split * 100)
    test = floor(test_split * 100)
    valid = floor(valid_split * 100)
    if (train + test + valid) != 100 or train_split < 0 or test_split < 0 or valid_split < 0:
        print("Invalid splitting.")
        return
    
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
    
    train_data_list = img_data_list[:floor(train_split * size)]
    train_img_list = img_name[:floor(train_split * size)]

    test_data_list = img_data_list[floor(train_split * size):floor(test_split * size)]
    test_img_list = img_name[floor(train_split * size):floor(test_split * size)]

    valid_data_list = img_data_list[floor(test_split * size): size - 1]
    valid_img_list = img_name[floor(test_split * size): size - 1]

    return train_data_list,test_data_list,valid_data_list,train_img_list,test_img_list,valid_img_list


def parse_fn(data_list, input_shape=(512,512)):
    x = tf.cast(data_list, tf.float32)
    x = x / 127.5 - 1
    return x