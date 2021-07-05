from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model('SVM/SVM_model/')
    return model

def read_image(file):
    image = Image.open(BytesIO(file))
    return image

def predict(image : np.array):
    model = load_model()
    predictions = model.predict(image)
    return predictions