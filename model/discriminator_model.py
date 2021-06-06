from generator_model import ConvTranBlock
from tensorflow import keras
from keras import layers

def ConvBlock(layer,filter_num, filter_size, stride, padding):
    layer = layers.Conv2D(filter_num, filter_size, strides=stride, padding=padding, use_bias='False')(layer)
    layer = layers.BatchNormailzation()(layer)
    layer = layers.LeakyReLU()(layer)
    return layer

def Discriminator(input_shape=(512,512,3), name='Discriminator'):
    #Default input shape:
    inputs = layers.Input(shape=input_shape)

    #First layer output shape: (256,256,64)
    l = layers.Conv2D(64,3,strides=2, padding='same')(inputs)
    l = layers.LeakyReLU()(l)

    #Second layer output shape: (128,128,128)
    l = ConvBlock(l,128,3,2,'same')

    #Third layer output shape: (32,32,256)
    l = ConvBlock(l,128,3,4,'same')

    #Fourth layer output shape: (4,4,512)
    l = ConvBlock(l,512,3,8,'same')

    #Final output shape: (1,1,1)
    outputs = layers.Conv2D(512,3,strides=4,padding='valid')(l)
    

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


