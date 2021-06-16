import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models


def Generator(input_shape=(1,1,128), name='Generator'):
    model = models.Sequential(name=name)

    #(1,1) -> (2,2)
    model.add(layer=layers.Conv2DTranspose(256,(2,2),strides=1, input_shape=input_shape, use_bias=False))
    model.add(layer=layers.BatchNormalization())
    model.add(layer=layers.LeakyReLU())

    #(2,2) -> (4,4)
    model.add(layer=layers.Conv2DTranspose(128,(3,3),strides=2,padding='same', use_bias=False))
    model.add(layer=layers.BatchNormalization())
    model.add(layer=layers.LeakyReLU())

    #(4,4) -> (8,8)
    model.add(layer=layers.Conv2DTranspose(128,(3,3),strides=2,padding='same', use_bias=False))
    model.add(layer=layers.BatchNormalization())
    model.add(layer=layers.LeakyReLU())

    #(8,8) -> (16,16)
    model.add(layer=layers.Conv2DTranspose(64,(3,3),strides=2,padding='same', use_bias=False))
    model.add(layer=layers.BatchNormalization())
    model.add(layer=layers.LeakyReLU())

    #(16,16) -> (64,64)
    model.add(layer=layers.Conv2DTranspose(3,(3,3),strides=4,padding='same', use_bias=False))
    model.add(layer=layers.BatchNormalization())
    model.add(layer=layers.LeakyReLU())

    model.add(layer=layers.Dense(3,activation=tf.nn.tanh))

    return model

