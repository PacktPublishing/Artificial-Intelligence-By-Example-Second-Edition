# -*- coding: utf-8 -*-
# Decision Tree Classifier
# Copyright 2019 Denis Rothman MIT License. See LICENSE in GitHub directory.
import pandas as pd #data processing
from sklearn.tree import DecisionTreeClassifier #the dt classifier
from sklearn.model_selection import train_test_split  #split the data into training data and testing data
from sklearn import metrics #measure prediction performance  
import pickle #save and load estimator models

#loading dataset
col_names = ['f1', 'f2','label']
df = pd.read_csv("ckmc.csv", header=None, names=col_names)

print(df.head())

#defining features and label (classes)
feature_cols = ['f1', 'f2']
X = df[feature_cols] # Features
y = df.label # Target variable

print(X)

print(y)

# splitting df (dataset) into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# create the decision tree classifier
dtc = DecisionTreeClassifier()

# train the decision tree
dtc = dtc.fit(X_train,y_train)

#predictions on X_test
print("prediction")
y_pred = dtc.predict(X_test)
print(y_pred)

# model accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#save model
pickle.dump(dtc, open("dt.sav", 'wb'))


''' Uncomment this part of the code to generate a graph and png
    set graph=1 to activate the function, graph=0 to deactivate the function
from sklearn import tree
import pydotplus

graph=1
if(graph==1):
  # Creating the graph and exporting it
  dot_data = tree.export_graphviz(dtc, out_file=None, 
                                  filled=True, rounded=True,
                                  feature_names=feature_cols,  
                                  class_names=['0','1','2','3','4','5'])

  #creating graph 
  graph = pydotplus.graph_from_dot_data(dot_data)  

  #save graph
  image=graph.create_png()
  graph.write_png("kmc_dt.png")
'''
