#K-means clustering - Mini-Batch-Shuffling
#Build with Sklearn
#Copyright 2018 Denis Rothman MIT License. See LICENSE.
from sklearn.cluster import KMeans  
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import numpy as np

#I.The training Dataset 
dataseti = pd.read_csv('data.csv')
print (dataseti.head())
print("initial order")
print(dataseti)

print("shuffled")
dataset=shuffle(dataseti, random_state=13)
print(dataset)

n=1000
dataset1=np.zeros(shape=(n,2))
for i in range (n):
    dataset1[i][0]=dataset.iloc[i,0]
    dataset1[i][1]=dataset.iloc[i,1]

print("shuffled selection")
print(dataset1)

#II.Hyperparameters
# Features = 2
k = 6
kmeans = KMeans(n_clusters=k)

#III.K-means clustering algorithm
kmeans = kmeans.fit(dataset1)         #Computing k-means clustering
gcenters = kmeans.cluster_centers_   # the geometric centers or centroids
print("The geometric centers or centroids:")
print(gcenters)

#IV.Defining the Result labels 
labels = kmeans.labels_
colors = ['blue','red','green','black','yellow','brown','orange']

#V.Displaying the results : datapoints and clusters
y = 0
for x in labels:
    plt.scatter(dataset1[y,0], dataset1[y,1],color=colors[x])
    y+=1       
for x in range(k):
    lines = plt.plot(gcenters[x,0],gcenters[x,1],'kx')    

title = ('No of clusters (k) = {}').format(k)
plt.title(title)
plt.xlabel('Distance')
plt.ylabel('Location')
plt.show()

#VI.Test dataset and prediction
x_test = [[40.0,67],[20.0,61],[90.0,90],
          [50.0,54],[20.0,80],[90.0,60]]
prediction = kmeans.predict(x_test)
print("The predictions:")
print (prediction)
