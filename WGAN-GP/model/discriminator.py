from tensorflow import keras
from keras import layers
from keras import models


def Discriminator(input_shape=(64,64,3), name='Discriminator'):
    model = models.Sequential(name=name)

    #(64,64) -> (32,32)
    model.add(layer=layers.Conv2D(64,(3,3), strides=2, padding='same', input_shape=input_shape))
    model.add(layer=layers.BatchNormalization())
    
    #(32,32) -> (16,16)
    model.add(layer=layers.Conv2D(128,(3,3),strides=2,padding='same'))
    model.add(layer=layers.BatchNormalization())
    model.add(layer=layers.LeakyReLU())

    #(16,16) -> (8,8)
    model.add(layer=layers.Conv2D(128,(3,3),strides=2,padding='same'))
    model.add(layer=layers.BatchNormalization())
    model.add(layer=layers.LeakyReLU())

    #(8,8) -> (4,4)
    model.add(layer=layers.Conv2D(256,(3,3),strides=2,padding='same'))
    model.add(layer=layers.BatchNormalization())
    model.add(layer=layers.LeakyReLU())

    #(4,4) -> (1,1)
    model.add(layer=layers.Conv2D(1,(4,4),strides=1))

    return model

