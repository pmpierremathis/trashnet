import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory



#default path colab
path= '/content/drive/MyDrive/trashnet/dataset_project/dataset_train'



img_width = 512
img_height = 384
colors = 3


def get_ds_train(directory=path):
    ds_train = image_dataset_from_directory(
      directory=path,
      labels='inferred',
      label_mode = 'categorical',
      batch_size=16,
      image_size=(img_height, img_width),
      shuffle=True,
      seed=123,
      validation_split=0.1,
      subset='training')
    return ds_train

def get_ds_val(directory=path):
    ds_val = image_dataset_from_directory(
        directory=path,
        labels='inferred',
        label_mode='categorical',
        batch_size=16,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset='validation',
    )
    return ds_val

path_test = '/content/drive/MyDrive/trashnet/dataset_project/dataset_test'

def get_ds_test(directory=path_test):
    ds_test = image_dataset_from_directory(
        directory=path_test,
        labels='inferred',
        label_mode='categorical',
        batch_size=16,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=123,
    )
    return ds_test
