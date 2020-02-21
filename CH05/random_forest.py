# -*- coding: utf-8 -*-
# Random Forest Classifier
# Copyright 2019 Denis Rothman MIT License. See LICENSE in GitHub directory.
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import pandas as pd
import numpy as np

pp=0 # print information
# load dataset
col_names = ['f1', 'f2','label']
df = pd.read_csv("ckmc.csv", header=None, names=col_names)

if pp==1:
    print(df.head())

#loading features and label (classes)
feature_cols = ['f1', 'f2']
X = df[feature_cols] # Features
y = df.label # Target variable

if pp==1:
    print(X)
    print(y)

#Divide the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Creating Random Forest Classifier and  training
clf = RandomForestClassifier(n_estimators=25,random_state=None,bootstrap=True)
clf.fit(X_train, y_train)

#Predictions
y_pred = clf.predict(X_test)

if pp==1:
    print("predictions:")
    print(y_pred)

#Metrics
ae=metrics.mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:',round(ae,3))


#Double Check the Model's Accuracy
doublecheck=1  # 1=yes, 0=no
if doublecheck==1:
    t=0
    f=0
    for i in range(0,1000):
        xf1=df.at[i,'f1'];xf2=df.at[i,'f2'];xclass=df.at[i,'label']
        X_DL = [[xf1,xf2]]
        prediction =clf.predict(X_DL)
        e=False
        if(prediction==xclass):
            e=True
            t+=1
        if(prediction!=xclass):
            e=False
            f+=1
        if pp==1:
            print (i+1,"The prediction for",X_DL," is:",str(prediction).strip('[]'),"the class is",xclass,"acc.:",e)

    acc=round(t/(t+f),3)
    print("true:",t,"false",f,"accuracy",acc)
    print("Absolute Error",round(1-acc,3))

