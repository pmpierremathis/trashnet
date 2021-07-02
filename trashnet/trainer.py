import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers, losses, regularizers
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.applications.resnet_v2 import ResNet50V2

from trashnet.get_data import get_ds_val, get_ds_train, get_ds_test
from trashnet.preprocess import augment, preprocess

img_width = 512
img_height = 384
colors = 3

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_val = get_ds_val()
ds_train = get_ds_train()
ds_test = get_ds_test()

ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE).map(
    preprocess, num_parallel_calls=AUTOTUNE)
ds_val = ds_val.map(preprocess, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE)

ds_train = ds_train.cache()
ds_train = ds_train.prefetch(AUTOTUNE)

ds_val = ds_val.prefetch(AUTOTUNE)
ds_test = ds_test.prefetch(AUTOTUNE)

# load model ResNet50V2
def load_model():
    model = ResNet50V2(weights='imagenet',
                       input_shape=(img_height, img_width, colors),
                       include_top=False,
                       input_tensor=None,
                       pooling=None,
                       classifier_activation='softmax')
    return model

# setting model non trainable
def set_nontrainable_layers(model):
    # Set the first layers to be untrainable
    model.trainable = False
    return model


# adding last layers trainable
def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainables, and add additional trainable layers on top'''
    base_model = set_nontrainable_layers(model)
    flattening_layer = layers.Flatten()
    dense_layer1 = layers.Dense(50, activation='relu')
    dropout1 = layers.Dropout(0.2)

    dense_layer2 = layers.Dense(20, activation='relu')
    dropout2 = layers.Dropout(0.2)

    prediction_layer = layers.Dense(6, activation='softmax')

    model = Sequential([
        base_model,
        flattening_layer,
        dense_layer1,
        dropout1,
        dense_layer2,
        dropout2,
        prediction_layer
    ])
    return model

# building model
def build_model_resnet():
    ### Load ResNet trained model
    model = load_model()

    ### Set ResNet untrainable + add last layers
    final_model = add_last_layers(model)

    ### Compile model
    opt = optimizers.Adam(learning_rate=1e-4)
    final_model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])

    return final_model


def fit_model():

    # get data
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    ds_val = get_ds_val()
    ds_train = get_ds_train()
    ds_test = get_ds_test()

    ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE).map(preprocess, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE)

    ds_train = ds_train.cache()
    ds_train = ds_train.prefetch(AUTOTUNE)

    ds_val = ds_val.prefetch(AUTOTUNE)
    ds_test = ds_test.prefetch(AUTOTUNE)

    #set model
    model_resnet = build_model_resnet()
    es = EarlyStopping(patience=15, restore_best_weights=True)
    history_resnet = model_resnet.fit(ds_train,
                                  validation_data=ds_val,
                                  epochs=1000,
                                  callbacks=[es])
    evaluation = model_resnet.evaluate(ds_test)
    return f'Accuracy score on test set : {evaluation}'
