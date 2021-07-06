from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

img_width = 512
img_height = 384
colors = 3
model_path = "model/ResNet_model/"

def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

def read_image(file):
    image = Image.open(BytesIO(file))
    image_array  = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.resize(image_array, (img_height, img_width))
    image = image / 255
    image = tf.expand_dims(image, axis = 0)
    return image
