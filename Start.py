# Import all the necessary files!
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from flask import Flask,render_template, request
import cv2
import json
import numpy as np
import base64
from datetime import datetime

app = Flask(__name__)

model = tf.keras.models.load_model('facenet_keras.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/doctor_login')
def dl():
    return render_template('doctor_login.html')

@app.route('/doctor_register')
def dr():
    return render_template('doctor_registration.html')

@app.route('/user_login')
def ul():
    return render_template('user_login.html')

@app.route('/user_registration')
def ur():
    return render_template('user_registration.html')




def img_to_encoding(path, model):
    img1 = cv2.imread(path, 1)
    img = img1[..., ::-1]
    dim = (160, 160)
    # resize image
    if (img.shape != (160, 160, 3)):
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    x_train = np.array([img])
    print(x_train)
    embedding = model.predict(x_train)
    # print("embedding",embedding)
    return embedding

def img_to_encoding(path, model):
  img1 = cv2.imread(path, 1)
  img = img1[...,::-1]
  dim = (160, 160)
  # resize image
  if(img.shape != (160, 160, 3)):
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  x_train = np.array([img])
  embedding2 = model.predict_on_batch(x_train)
  return embedding2

database = {}
database["SRK"] = img_to_encoding("Sharukh.jpg", model)
database["Salman"] = img_to_encoding("salman.jpg", model)
database["Akshay"] = img_to_encoding("akshay.jpg", model)


def verify(image_path, identity, database, model):
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding - database[identity])
    print(dist)
    if dist < 20:
        print("It's " + str(identity) + ", welcome in!", dist)
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away", dist)
        door_open = False
    return dist, door_open


verify("SRK.jpg", "SRK", database, model)
verify("Sharukh.jpg", "Akshay", database, model)







if __name__ == "__main__":
    app.run(debug=True)