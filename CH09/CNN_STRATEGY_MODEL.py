#Convolutional Neural Network (CNN) : training and saving the model
#Built with Tensorflow 2
#Copyright 2019 Denis Rothman MIT License. READ LICENSE.
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

A=['dataset_O/','dataset_traffic/','dataset/']     
scenario=2            #reference to A
directory=A[scenario] #transfer learning parameter (choice of images)
print("directory",directory)

# Part 1 - Building the CNN

#Training Scenarios
estep=1000 #8000->100
batchs=10  #32->10
vs=100     #2000->100
ep=3       #25->2

# Initializing the CNN
print("Step 0 Initializing")
classifier =  models.Sequential()

# Step 1 - Convolution
print("Step 1 Convolution")
classifier.add(layers.Conv2D(32, (3, 3),input_shape = (64, 64, 3), activation = 'relu'))
 
# Step 2 - Pooling
print("Step 2 MaxPooling2D")
classifier.add(layers.MaxPooling2D(pool_size = (2, 2)))

# Step 3 Adding a second convolutional layer and pooling layer
print("Step 3a Convolution")
classifier.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

print("Step 3b Pooling")
classifier.add(layers.MaxPooling2D(pool_size = (2, 2)))

# Step 4 - Flattening
print("Step 4 Flattening")
classifier.add(layers.Flatten())

# Step 4 - Full connection
print("Step 5 Dense")
classifier.add(layers.Dense(units = 128, activation = 'relu'))
classifier.add(layers.Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
print("Step 6 Optimizer")
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#print("CNN Structure Summary")
classifier.summary()

# Part 2 - Fitting the CNN to the images

print("Step 7a train")
train_datagen = tf.compat.v2.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

print("Step 7b training set")
training_set = train_datagen.flow_from_directory(directory+'training_set',
                                                 target_size = (64, 64),
                                                 batch_size = batchs,
                                                 class_mode = 'binary')

print("Step 8a test")
test_datagen = tf.compat.v2.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)


print("Step 8b testing set")
test_set = test_datagen.flow_from_directory(directory+'test_set',
                                            target_size = (64, 64),
                                            batch_size = batchs,
                                            class_mode = 'binary')
print("Step 9 training")
print("Classifier",classifier.fit_generator(training_set,
                         steps_per_epoch = estep,
                         epochs = ep,
                         validation_data = test_set,
                         validation_steps = vs,verbose=2))

print("Step 10: Saving the model")
classifier.save(directory+"model/model3.h5")
print("Training over, model saved")
