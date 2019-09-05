#Convolutional Neural Network (CNN) : loading and running a trained model
#Built with Tensorflow 2
#Copyright 2019 Denis Rothman MIT License. READ LICENSE.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

#Directory
directory='dataset/' 
print("directory",directory)

#____________________LOAD MODEL____________________________

loaded_model = keras.models.load_model(directory+"model/model3.h5")
print(loaded_model.summary())
# __________________compile loaded model
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#___________________ LOAD WEIGHTS_________________________

print("GLOBAL MODEL STRUCTURE")
print(loaded_model.summary())

print("DETAILED MODEL STRUCTURE")
for layer in loaded_model.layers:
    print(layer.get_config())
 
print("WEIGHTS")
for layer in loaded_model.layers:
     weights = layer.get_weights() # list of numpy arrays
     print(weights)
    
     
