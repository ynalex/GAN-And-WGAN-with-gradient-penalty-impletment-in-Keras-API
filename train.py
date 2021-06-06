from numpy.core.fromnumeric import shape
import tensorflow as tf
import os 
import cv2
from math import floor

@tf.function
def train_generator(generator, discriminator, generator_optimizer, batch_size=64, z_dim=128):
    with tf.GradientTape() as tape:
        tape.watch(generator.trainable_variable)
        random_vector = tf.random.normal(shape=(batch_size,1,1,z_dim))
        fake_image = generator(random_vector, training=True)

        fake_logit = discriminator(fake_image, training=True)

        g_loss = generator_loss(fake_logit)

    gradients = tape.graident(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return g_loss

@tf.functon
def train_discriminator(generator, discriminator, discriminator_optimizer, real_image, batch_size=64, z_dim=128):
    with tf.GradientTape() as tape:
        random_vector = tf.random.normal(shape=(batch_size,1,1,z_dim))
        fake_image = generator(random_vector, training=True)

        real_logit = discriminator(real_image, training=True)
        fake_logit = discriminator(fake_image, training=True)

        real_loss, fake_loss = discriminator_loss(real_logit, fake_logit)

        d_loss = real_loss + fake_loss
    
    gradient = tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradient, discriminator.trainable_variables))
    return d_loss

def train_gan(
    generator, discriminator,
    generator_optimizer, discriminator_optimizer, 
    train_data_list,
    train_ratio = 6,
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
        for step, real_image in enumerate(train_data_list):
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
        if generator_optimizer.iterations.numpy() % 16 == 0:
            fake = generator(random_vector, training=False)
            save_image = save_generated_img(fake)
            save_image_dir = os.path.join(generated_image_dir, 'generated_image_{}.jpg'.format(floor(generator_optimizer.iterations.numpy()/16)))
            cv2.imwrite(save_image_dir, save_image)
            
        #saving generator and discrminator weight
        if epochs != 0:
            generator.save_weights(os.path.join(g_model_dir, 'geneartor_epoch_{}.h5'.format(epochs)))
            discriminator.save_weights(os.path.join(d_model_dir, 'discriminator_epoch_{}.h5'.format(epochs)))