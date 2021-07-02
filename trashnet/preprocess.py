import pandas as pd
import numpy as np

import tensorflow as tf


def preprocess(image, label):
    image = image / 255
    return image, label


def augment(image, label):
    if np.random.rand(1) < 0.2:
        image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label
