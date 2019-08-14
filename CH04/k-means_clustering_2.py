#K-means clustering
#Build with Sklearn
#Copyright 2019 Denis Rothman MIT License. See LICENSE.

from sklearn.cluster import KMeans  
import pandas as pd
from matplotlib import pyplot as plt
import pickle

#load model
filename="kmc_model.sav"
kmeans = pickle.load(open(filename, 'rb'))

#test data
x_test = [[40.0,67],[20.0,61],[90.0,90],
          [50.0,54],[20.0,80],[90.0,60]]

#prediction
prediction = kmeans.predict(x_test)
print("The predictions:")
print (prediction)

