from tensorflow import keras
from keras import layers
from keras import models


def Discriminator(input_shape=(256,256,3), name='Discriminator'):
    model = models.Sequential(name=name)

    #(256,256) -> (128,128)
    model.add(layer=layers.Conv2D(64,(3,3), strides=2, padding='same', input_shape=input_shape))
    model.add(layer=layers.BatchNormalization())
    
    #(128,128) -> (64,64)
    model.add(layer=layers.Conv2D(128,(3,3),strides=2,padding='same'))
    model.add(layer=layers.BatchNormalization())
    model.add(layer=layers.LeakyReLU())

    #(64,64) -> (16,16)
    model.add(layer=layers.Conv2D(256,(3,3),strides=4,padding='same'))
    model.add(layer=layers.BatchNormalization())
    model.add(layer=layers.LeakyReLU())

    #(16,16) -> (4,4)
    model.add(layer=layers.Conv2D(512,(3,3),strides=4,padding='same'))
    model.add(layer=layers.BatchNormalization())
    model.add(layer=layers.LeakyReLU())

    #(4,4) -> (1,1)
    model.add(layer=layers.Conv2D(1,(4,4),strides=1))

    return model

