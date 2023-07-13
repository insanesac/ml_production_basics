#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:53:41 2023

@author: insanesac
"""

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub

import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = r'/home/insanesac/.keras/datasets/flower_photos'

class_name = os.listdir(data_dir)
BATCH_SIZE = 16
IMG_SIZE = (224, 224)

class_names = os.listdir(data_dir)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(factor=0.15),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    tf.keras.layers.RandomContrast(factor=0.1),
])

# def process_path(file_path):
#     # load the raw data from the file as a string
#     img = tf.io.read_file(file_path)
#     # convert the compressed string to a 3D uint8 tensor
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, IMG_SIZE)
#     # create one hot labels from class names
#     parts = tf.strings.split(file_path, os.sep)
#     one_hot = parts[-2] == class_names
#     return img/255.0, tf.argmax(tf.cast(one_hot, tf.int32))

def load_images(imagePath):
	# read the image from disk, decode it, convert the data type to
	# floating point, and resize it
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    # parse the class label from the file path
    label = tf.strings.split(imagePath, os.path.sep)
    one_hot = label[-2] == class_names
    # return the image and the label
    return (image, tf.argmax(tf.cast(one_hot, tf.int32)))

# train_dataset, validation_dataset = keras.utils.split_dataset(dataset, left_size=0.8)
train_list_ds  = tf.data.Dataset.list_files(str(data_dir)+'/*/*')
train_list_ds = train_list_ds.shuffle(len(train_list_ds))
print("Total file: ", len(train_list_ds))

# split the data set to train and validation set
train_size = int(len(train_list_ds) * 0.8) 
val_size = int(len(train_list_ds) * 0.2)

train_ds = train_list_ds.take(train_size)
val_ds = train_list_ds.skip(train_size)

print('Train files: ', len(train_ds))
print('Validation files: ', len(val_ds))

# this will help to delegate the decision on the level of parallelism to optimize the CPU/GPU utilization.
AUTOTUNE = tf.data.AUTOTUNE

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(load_images, num_parallel_calls=AUTOTUNE)#.map(augment_using_ops, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(load_images, num_parallel_calls=AUTOTUNE)

# ### Prefetching in tf.data allows the preprocessing of the data and model execution of a training step to overlap.
def configure_for_performance(ds,augment=False):
    ds = ds.cache()                          # Cache a dataset, either in memory or on local storage
    ds = ds.batch(16)                # Create batches of images
    ds = ds.prefetch(buffer_size=AUTOTUNE)   # Loads image inputs for the next epoch
    return ds

efficientnet_v2 = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2"
inception_v3 = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

feature_extractor_model = efficientnet_v2 #@param ["mobilenet_v2", "inception_v3"] {type:"raw"}

feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model,
    input_shape=(224, 224, 3),
    trainable=False)

X_train = configure_for_performance(train_ds, augment=True)
X_val = configure_for_performance(val_ds)

# IMG_SHAPE = IMG_SIZE + (3,)
# base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
#                                                 include_top=False,
#                                                 weights='imagenet')

# base_model.trainable = False

# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# prediction_layer = tf.keras.layers.Dense(5)

# inputs = tf.keras.Input(shape=(299, 299, 3))
# x = base_model(inputs, training=False)
# x = global_average_layer(x)
# # x = tf.keras.layers.Dropout(0.2)(x)
# outputs = prediction_layer(x)
# model = tf.keras.Model(inputs, outputs)
model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(5)
])

model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

initial_epochs = 50

history = model.fit(X_train,
                    epochs=initial_epochs,
                    validation_data=X_val)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

saved_model_path = "./model/"
model.save(saved_model_path)