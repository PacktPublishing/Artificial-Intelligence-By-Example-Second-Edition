#Naive Bayes applied to Blockchains
#Built with slearn.naive_bayes
#Copyright 2019 Denis Rothman MIT License. See LICENSE.

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


#Reading the data
df = pd.read_csv('data_BC.csv')
print("Blocks of the Blockchain")
print (df.head())

# Prepare the training set
X = df.loc[:,'DAY':'BLOCKS']
Y = df.loc[:,'DEMAND']

#Choose the class
clfG = GaussianNB()
# Train the model
clfG.fit(X,Y)

# Predict with the model(return the class)
print("Blocks for the prediction of the A-F blockchain")
blocks=[[14,1345,12],
        [29,2034,50],
        [30,7789,4],
        [31,6789,4]]
print(blocks)
prediction = clfG.predict(blocks)
for i in range(4):
    print("Block #",i+1," Gauss Naive Bayes Prediction:",prediction[i])


