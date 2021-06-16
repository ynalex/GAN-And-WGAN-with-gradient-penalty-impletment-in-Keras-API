import tensorflow as tf
import numpy as np

def interpolate(x, y):
    shape = [tf.shape(x)[0]] + [1] * (x.shape.ndims - 1)
    print(shape)
    epsilon = tf.random.uniform(shape = shape, minval=0., maxval=1.)
    inter = (epsilon * x) + (1 - epsilon) * y
    #inter.set_shape(shape)
    return inter 

def generator_loss(fake_logit):
    g_loss = -tf.reduce_mean(fake_logit)
    return g_loss

def discriminator_loss(real_logit, fake_logit):
    real_loss = -tf.reduce_mean(real_logit)
    fake_loss = tf.reduce_mean(fake_logit)

    return real_loss, fake_loss

def gradient_penalty(discriminator, real_img, fake_img):
    def interpolate(x, y):
        #shape = [tf.shape(x)[0]] + [1] * (x.shape.ndims - 1)
        epsilon = tf.random.uniform(shape = x.shape, minval=0., maxval=1.)
        inter = (epsilon * x) + (1 - epsilon) * y
        #inter.set_shape(shape)
        return inter 
    x = interpolate(real_img, fake_img)
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred_logit = discriminator(x)
    grad = tape.gradient(pred_logit, x)
    gp = tf.reduce_mean((tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)-1.)**2)
    return gp
"""

Ref: https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py
"""
