import enum
import tensorflow as tf
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

AUTOTUNE = tf.data.experimental.AUTOTUNE

def _get_image_name(path, image_count):
    filename = []
    count = 0
    for file in sorted(os.listdir(path)):
        if count == image_count:
            break
        filename.append(os.path.join(path,file))
        count += 1
    #filename = tf.constant(filename)
    return filename
def preprocessing(im):
    im = tf.image.decode_jpeg(im, channels=3)
    im = tf.image.resize(im, [64,64])
    im/=127.5 - 1
    return im

def load_and_preprocessing(filename):
    im = tf.io.read_file(filename)
    return preprocessing(im)

def optimize_ds(ds, image_count, batch_size, epochs):
    ds = ds.shuffle(buffer_size=image_count)
    ds = ds.repeat(30)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def create_train_test_split(path, train_split, test_split, image_count, batch_size, epochs):
    if int(train_split + test_split) != 1:
        print("Invalid inputs.")
        return
    filename = _get_image_name(path, image_count)

    train_filename = filename[:int(train_split*image_count)]
    test_filename = filename[int(train_split*image_count):]

    train_ds = tf.data.Dataset.from_tensor_slices(train_filename)
    test_ds = tf.data.Dataset.from_tensor_slices(test_filename)

    train_ds = train_ds.map(load_and_preprocessing, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(load_and_preprocessing, num_parallel_calls=AUTOTUNE)

    train_ds = optimize_ds(train_ds, image_count, batch_size, epochs)
    test_ds = optimize_ds(test_ds, image_count, batch_size, epochs)

    return train_ds, test_ds
