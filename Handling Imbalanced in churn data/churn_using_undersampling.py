# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:21:36 2020

@author: degananda.reddy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
df=pd.read_csv(r"Churn_Modelling.csv")
df.isnull().any().sum()
df.dtypes
df.drop(['Surname','CustomerId','RowNumber'],axis=1,inplace=True)
df['Geography'].value_counts()
geography=pd.get_dummies(df['Geography'],drop_first=True)
gender=pd.get_dummies(df['Gender'],drop_first=True)
df=pd.concat([df,geography,gender],axis=1)
df.drop(['Geography','Gender'],axis=1,inplace=True)
df.dtypes
y=df['Exited']
X=df.drop(['Exited'],axis=1)
#under sampling
from imblearn.under_sampling import NearMiss
nm=NearMiss()
x_res,y_res=nm.fit_sample(X,y)
from collections import Counter
Counter(y_res)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier=Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))
# Adding the second hidden layer
classifier.add(Dense(units = 3, kernel_initializer = 'he_uniform',activation='relu'))                                            
classifier.add(Dense(units = 2, kernel_initializer = 'he_uniform',activation='relu'))                                            

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set   
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 100)
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
