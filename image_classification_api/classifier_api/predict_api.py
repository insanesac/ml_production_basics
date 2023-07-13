#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:44:47 2023

@author: insanesac
"""

import tensorflow as tf
from flasgger import Swagger
from flask import Flask, request
from tensorflow import keras
from PIL import Image
app = Flask(__name__)
swagger = Swagger(app)
saved_model_path = "./model/"
model = keras.models.load_model(saved_model_path)

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def load_images(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    return image/255.0
            
@app.route('/predict', methods=['POST'])
def predict():
    """ Example endpoint returning prediction of flowers
    ---
    parameters:
      - name: image
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Prediction
    """
    image = Image.open(request.files['image'])
    image = tf.expand_dims(load_images(image), axis=0)
    prediction = model.predict(image)
    prediction = tf.math.argmax(prediction, axis=-1)
    label = class_names[prediction.numpy()[0]]
    return label

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

