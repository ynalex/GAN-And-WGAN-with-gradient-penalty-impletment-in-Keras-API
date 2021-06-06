from tensorflow import keras
from keras import layers


def ConvTranBlock(layer,filter_num, filter_size, stride, padding):
    layer = layers.Conv2dTranspose(filter_num, filter_size, strides = stride, padding=padding, use_bias=False)(layer)
    layer = layers.BatchNormalization()(layer)
    layer = layers.LeakyReLU()(layer)
    return layer

def Generator(input_shape = (1,1,128), name = "Genartor"):
    #Dedault input shape : (1,1,128)
    inputs = keras.Input(shape=input_shape)

    #First layer output shape: (4,4,512)
    l = layers.Conv2dTranspose(512, 4, strides = 1, padding='valid', use_bias=False)(inputs)
    l = layers.BatchNormalization()(l)
    l = layers.LeakyReLU()(l)

    #Second layer output shape: (8,8,256) 
    l = ConvTranBlock(l,256,3,2,'same')

    #Third layer output shape: (32,32,128)
    l = ConvTranBlock(l,128,3,4,'same')

    #Fourth layer output shape: (128,128,64)
    l = ConvTranBlock(l,64,3,4,'same')

    #Fifth layer output shape: (512,512,3)
    l = ConvTranBlock(l,3,3,4,'same')

    #Final output shape: (512,512,3)
    outputs = layers.Activation('tanh')(l)

    return keras.Mode(inputs=inputs, outputs=outputs, name=name)

"""

Ref: Calcualting output shape: https://datascience.stackexchange.com/questions/26451/how-to-calculate-the-output-shape-of-conv2d-transpose

"""