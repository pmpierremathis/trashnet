import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf


def plot_history(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label='train' + exp_name)
    ax1.plot(history.history['val_loss'], label='val' + exp_name)
    #ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
    #ax2.set_ylim(0.25, 1.)
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)


def get_ys(ds_test, model):
    """Function that return y_test and y_pred from ds_test and a trained model"""
    y_pred = None
    y_true = None
    for images, labels in ds_test:
        # Creating y_true
        if y_true is None:
            y_true = labels
        else:
            y_true = tf.concat([y_true, labels], axis=0)

        # Creating y_pred
        if y_pred is None:
            y_pred = model.predict(images)
        else:
            y_pred = tf.concat([y_pred, model.predict(images)], axis=0)

    # Removing the one hot encoding
    y_true_arg = np.argmax(y_true.numpy(), axis=1)
    y_pred_arg = np.argmax(y_pred.numpy(), axis=1)

    return y_true_arg, y_pred_arg


def get_confusion_matrix(ds_test, model):
    """Function that return the confusion matrix from ds_test and a trained model"""
    target_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    # obtain y_true and y_prd
    y_true, y_pred = get_ys(ds_test, model)

    #creating confusion matrix
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=6).numpy()
    print(cm)
    # normalization of the matrix
    cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    print(cm_norm)
    #creation of a Dataframe
    cm_df = pd.DataFrame(cm_norm,target_names, target_names)
    print(cm_df)
    return cm_df


def plot_confusion_matrix(cm_df):
    """Function to plot the DF of the confusion matrix from the confusion matrix dataframe"""
    plt.figure(figsize=(10,8))
    return sns.heatmap(cm_df, annot=True,cmap="YlGnBu")
