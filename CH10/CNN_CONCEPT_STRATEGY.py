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

#SCENARIO
directory='dataset/'
print("directory",directory)
display=1     #display images     
#MS1 = message for prediction=0, MS2=message for prediction=1
MS1='productive'
MS2='gap'

#____________________LOAD MODEL____________________________
loaded_model = keras.models.load_model(directory+"model/model3.h5")
print(loaded_model.summary())
# __________________compile loaded model
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#___________________IDENTIFY IMAGE FUNCTION_____________________

def identify(target_image):
    filename = target_image
    original = load_img(filename, target_size=(64, 64))
    if(display==1):
        plt.imshow(original)
        plt.show()
        
    numpy_image = img_to_array(original)
    inputarray = numpy_image[np.newaxis,...] # extra dimension to fit model
    arrayresized=np.resize(inputarray,(64,64))   
#___________________PREDICTION___________________________
    prediction = loaded_model.predict_proba(inputarray)
    return prediction
#___________________SEARCH STRATEGY_____________________
s=identify(directory+'classify/img1.jpg')
print("Prediction image 1",s)
if (int(s)==0):
    print('Classified in class A')
    print(MS1)
    
print('Seeking...')

s=identify(directory+'classify/img2.jpg')
print("Prediction image 2",s)
if (int(s)==1):
    print('Classified in class B')
    print(MS2)
         
