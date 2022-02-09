import os
import sys

# Flask
from flask import Flask,request, render_template, jsonify

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Some utilites
import numpy as np
import cv2
from util import base64_to_pil
from keras.models import model_from_json

# Declare a flask app
app = Flask(__name__)

# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

#
# MODEL_PATH = 'models/resnet_model.h5'
# model = load_weights(MODEL_PATH)
# model._make_predict_function()



#
# print('Model loaded. Check http://127.0.0.1:5000/')

MODEL_json_PATH = 'models/resnet_model.json'
MODEL_PATH = 'models/resnet_model.h5'

# json_file = open(MODEL_json_PATH, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
model = tf.keras.models.load_model(MODEL_PATH)
#model._make_predict_function()

print('Model loaded. Start serving...')

img_height,img_width=180,180
def model_predict(test_image, model):
    image = cv2.imread(test_image)
    image_resized = cv2.resize(image, (img_height, img_width))
    image = np.expand_dims(image_resized, axis=0)

    preds = model.predict(image)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        # Save the image to ./uploads
        img.save("./uploads/image.png")
        img_location = "./uploads/image.png"
        # Make prediction
        preds = model_predict(img_location, model)

        # Process your result for human
        output_pred_class = class_names[np.argmax(preds)]
        #print(output_pred_class)
        return jsonify(result=output_pred_class)

    return None


if __name__ == '__main__':
    app.run()