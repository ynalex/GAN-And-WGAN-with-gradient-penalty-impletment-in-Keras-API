import numpy as np
from numpy.lib.type_check import real
import tensorflow as tf
import os 
import cv2
from math import floor
from loss.loss import generator_loss, discriminator_loss
from model.discriminator import Discriminator
from model.generator import Generator
from utils.general import save_generated_image

@tf.function
def train_generator(generator, discriminator, generator_optimizer, batch_size=64, z_dim=128):
    with tf.GradientTape() as tape:
        
        random_vector = tf.random.normal(shape=(batch_size,1,1,z_dim))
        fake_image = generator(random_vector, training=True)

        fake_logit = discriminator(fake_image, training=True)

        g_loss = generator_loss(fake_logit)

    gradients = tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return g_loss

@tf.function
def train_discriminator(generator, discriminator, discriminator_optimizer, real_image, batch_size=64, z_dim=128):
    with tf.GradientTape() as tape:
        
        random_vector = tf.random.normal(shape=(batch_size,1,1,z_dim))
        fake_image = generator(random_vector, training=True)

        real_logit = discriminator(real_image, training=True)
        fake_logit = discriminator(fake_image, training=True)

        real_loss, fake_loss = discriminator_loss(real_logit, fake_logit)

        d_loss = real_loss + fake_loss
    
    gradients = tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    return d_loss

def train_gan(
    generator, discriminator,
    generator_optimizer, discriminator_optimizer, 
    train_data_list,
    train_ratio=6,
    batch_size=64, z_dim=128, epochs=30):
    
    #save models and model logs
    log_dir = "gan_log"
    g_model_dir = os.path.join(log_dir, 'generator_models')
    d_model_dir = os.path.join(log_dir, 'discriminator_models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(g_model_dir, exist_ok=True)
    os.makedirs(d_model_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(log_dir)

    #generate fake images when training
    random_vector = tf.random.normal(shape=(16,1,1,z_dim))
    generated_image_dir = os.path.join(log_dir, 'generated_image_while_training')
    os.makedirs(generated_image_dir, exist_ok=True)
    
    for iters in range(epochs):
        print('Cureently at epoch {}.'.format(iters + 1))
        for step, real_image in enumerate(train_data_list):

            """real_image = tf.reshape(real_image, (1,256,256,3))
            real_image = cv2.imread(os.path.join('datas/Image', real_image),cv2.COLOR_RGB2BGR)
            real_image = np.array(real_image)
            print(real_image.shape)
            real_image = normalize(real_image)
            real_image = tf.reshape(real_image, (1,256,256,3))"""
            
            #training discriminator
            d_loss = train_discriminator(generator, discriminator, discriminator_optimizer, real_image, batch_size, z_dim)

            #saving losses to TensorBoard log
            with summary_writer.as_default():
                tf.summary.scalar('discriminator_loss', d_loss, discriminator_optimizer.iterations)

            if discriminator_optimizer.iterations.numpy() % train_ratio == 0:
                #training generator
                g_loss = train_generator(generator, discriminator, generator_optimizer, batch_size, z_dim)

                #saving losses to TensorBoard log
                with summary_writer.as_default():
                    tf.summary.scalar('generator_loss', g_loss, generator_optimizer.iterations)
        
        #save generated image from generator when the generated is trained for 16 times

        random_vector_img = tf.random.normal(shape=(1,1,1,z_dim))
        fake = generator(random_vector_img, training=False)
        fake = np.squeeze(fake, axis=0)
        fake = fake * 255
        print(fake.shape)
        save_image_dir = os.path.join(generated_image_dir, "generated_image_{}.jpg".format(iters + 1))
        cv2.imwrite(save_image_dir, fake)
        """fake = generator(random_vector, training=False)
        save_image = save_generated_image(fake)
        save_image_dir = os.path.join(generated_image_dir, 'generated_image_{}.jpg'.format(iters))
        cv2.imwrite(save_image_dir, save_image)"""
            
        #saving generator and discrminator weight
        if epochs != 0:
            generator.save_weights(os.path.join(g_model_dir, 'geneartor_epoch_{}.h5'.format(epochs)))
            discriminator.save_weights(os.path.join(d_model_dir, 'discriminator_epoch_{}.h5'.format(epochs)))
    
""" random_vector = tf.random.normal(shape=(1,1,1,z_dim))
    fake = generator(random_vector, training=False)
    fake = np.array(fake)
    fake = np.squeeze(fake, axis=0)
    fake = fake * 255
"""