#Convolutional Neural Network (CNN) : Visualizing layer activity
#Built with Tensorflow 2
#Copyright 2020 Denis Rothman MIT License. READ LICENSE.
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
print(tf.__version__)
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import imageio as im
from pathlib import Path
import cv2
import glob

cv_img=[]
images = []
for img_path in glob.glob('dataset/training_set/img/*.png'):
    images.append(mpimg.imread(img_path))
    
plt.figure(figsize=(20,20)) #20,10
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)

#initialising the Tensorflow 2 classifier
classifier = models.Sequential()

#adding the convolution layers to the layers 
classifier.add(layers.Conv2D(32, (3, 3), padding='same', input_shape = (28, 28, 3), activation = 'relu'))
classifier.add(layers.Conv2D(32, (3, 3), activation='relu'))

#adding the max pooling layer to the layers
classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))

#adding the dropout layer to the layers
classifier.add(layers.Dropout(0.5)) # antes era 0.25

#adding more convolution layers to the layers
classifier.add(layers.Conv2D(64, (3, 3), padding='same', activation = 'relu'))
classifier.add(layers.Conv2D(64, (3, 3), activation='relu'))

#adding the max pooling layer to the layers
classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))

#adding the dropout layer to the layers
classifier.add(layers.Dropout(0.1))

#adding more convolution layers to the layers
classifier.add(layers.Conv2D(64, (3, 3), padding='same', activation = 'relu'))
classifier.add(layers.Conv2D(64, (3, 3), activation='relu'))

#adding a max pooling layer to the layers
classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))

#adding a dropout layer
classifier.add(layers.Dropout(0.1)) 

#adding flattening layer
classifier.add(layers.Flatten())

#adding dense-dropout-dense layers
classifier.add(layers.Dense(units = 512, activation = 'relu'))
classifier.add(layers.Dropout(0.1)) 
classifier.add(layers.Dense(units = 3, activation = 'relu'))

#Printing the model summary
print("Model Summary",classifier.summary())


# Compiling the convolutional neural network (CNN)
classifier.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (28,
                                                 28),
                                                 batch_size = 16,
                                                 class_mode =
                                                     'categorical')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (28, 28),
                                            batch_size = 16,
                                            class_mode =
                                                 'categorical')


#DISPLAYING EACH LAYER'S ACTIVITY


from keras.preprocessing import image
#Selecting an image for the activation model
img_path = 'dataset/test_set/img/img1.png'
img1 = image.load_img('dataset/test_set/img/img1.png', target_size=(28, 28))
img = image.img_to_array(img1)
img = np.expand_dims(img, axis=0)
img /= 255.
plt.imshow(img[0])
plt.show()
print("img tensor shape",img.shape)

#Selecting the number of layers to display
e=12 #last layer
layer_outputs = [layer.output for layer in classifier.layers[0:e]]

# Extracting the information of the top n layers
activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs)

# Activating the model
activations = activation_model.predict(img)

#layer names
layer_names = []
for layer in classifier.layers[:12]:
    layer_names.append(layer.name) 
    
images_per_row = 16

# Processing the layer outputs
for layer_name, layer_activation in zip(layer_names, activations): #getting the layer_names and their activations
    n_features = layer_activation.shape[-1]                        #features in the layer                    
    size = layer_activation.shape[1]                               #shape of the feature map
    n_cols = n_features // images_per_row                          #number of images per row
    display_grid = np.zeros((size * n_cols, images_per_row * size))#size of the display grid
    for col in range(n_cols):                                      #organizing the columns...
        for row in range(images_per_row):                          #...and rows to display
            image = layer_activation[0,:, :,col * images_per_row + row] #retrieving the image...
            image -= image.mean()                                       #...and processing it in the...
            if(image.std()>0):                                          #...following lines to display it
                image /= image.std()                                    
                image *= 64 
                image += 128 
                image = np.clip(image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, 
                         row * size : (row + 1) * size] = image

#displaying the layer names and processed grids
    print("Displaying layer:",layer_name) 
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig("dataset/output/"+layer_name)                    #saving the figures for further use
    plt.show()







