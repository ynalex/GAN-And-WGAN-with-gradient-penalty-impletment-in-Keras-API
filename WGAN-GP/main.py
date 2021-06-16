from datas.data import create_train_test_split
from model.discriminator import Discriminator
from model.generator import Generator
from model.train import train_gan
import tensorflow as tf
from tensorflow import keras
import numpy as np

#loading data and data preprocessing

img_folder_path = 'datas/Image'
test_split = 0.3
train_split = 0.7
size = 5000
z_dim = 128
batch_size = 30
lr = 0.0001
train_ratio = 5                                                                                                                                                                                                                                                                                 
epochs = 30


train_data_list, test_data_list = create_train_test_split(img_folder_path, train_split, test_split, size, batch_size, epochs)
#setting hyperparameter

#building models and optimizers

generator = Generator((1,1,z_dim))
discriminator = Discriminator((256,256,3))

g_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
d_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)

#train model
train_gan(generator, discriminator, g_optimizer, d_optimizer,train_data_list, train_ratio, batch_size, z_dim, epochs)

