#Chaining a K-means clustering algorithm to a Decision Tree
#Copyright 2019 Denis Rothman MIT License. See LICENSE.
from sklearn.cluster import KMeans  
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 

#----- chained K-means clustering and Decision Tree  
#I.KMC. The prediction dataset and model 
dataset = pd.read_csv('data.csv')
kmeans = pickle.load(open('kmc_model.sav', 'rb'))
   
#saveing the predictions
kmcpred=np.zeros((1000, 3)) 
predict=1
if predict==1:
    for i in range(0,1000):
        xf1=dataset.at[i,'Distance'];xf2=dataset.at[i,'location'];
        X_DL = [[xf1,xf2]]
        prediction = kmeans.predict(X_DL)
        #print (i+1,"The prediction for",X_DL," is:",str(prediction).strip('[]'))
        #print (i+1,"The prediction for",str(X_DL).strip('[]')," is:",str(prediction).strip('[]'))
        p=str(prediction).strip('[]')
        p=int(p)
        kmcpred[i][0]=int(xf1);kmcpred[i][1]=int(xf2);kmcpred[i][2]=p;
np.savetxt('ckmc.csv', kmcpred, delimiter=',', fmt='%d')

#II.Decison Tree.
adt=1 #activate decision tree or not
if adt==1:
    #I.DT. The prediction dataset and model 
    col_names = ['f1', 'f2','label']
    # load dataset
    ds = pd.read_csv('ckmc.csv', header=None, names=col_names)

    #split dataset in features and target variable
    feature_cols = ['f1', 'f2']
    X = ds[feature_cols] # Features
    y = ds.label # Target variable

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

    # Load model
    dt = pickle.load(open('dt.sav', 'rb'))

    #Predict the response for test dataset
    y_pred = dt.predict(X_test)

    # Model Accuracy
    acc=metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:",round(acc,3))


    #Double Check the Model's Accuracy
    doublecheck=1 #0 deactivated, 1 activated 
    if doublecheck==1:
        t=0
        f=0
        for i in range(0,1000):
            xf1=ds.at[i,'f1'];xf2=ds.at[i,'f2'];xclass=ds.at[i,'label']
            X_DL = [[xf1,xf2]]
            prediction =dt.predict(X_DL)
            e=False
            if(prediction==xclass):
                e=True
                t+=1
            if(prediction!=xclass):
                e=False
                f+=1
            print (i+1,"The prediction for",X_DL," is:",str(prediction).strip('[]'),"the class is",xclass,"acc.:",e)

        print("true:",t,"false",f,"accuracy",round(t/(t+f),3))

