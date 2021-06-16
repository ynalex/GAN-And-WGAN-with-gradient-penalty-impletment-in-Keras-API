import tensorflow as tf


def generator_loss(fake_logit):
    g_loss = tf.reduce_mean(tf.math.log(1-tf.sigmoid(fake_logit)))
    return g_loss

def discriminator_loss(real_logit, fake_logit):
    real_loss = -tf.reduce_mean(tf.math.log(tf.sigmoid(real_logit)))
    fake_loss = -tf.reduce_mean(tf.math.log(1-tf.sigmoid(fake_logit)))

    return real_loss, fake_loss

"""
Ref: https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py
"""