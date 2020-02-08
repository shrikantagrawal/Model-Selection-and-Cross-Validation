# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:12:47 2020

@author: Shrikant Agrawal
"""


# Model Selection and Cross Validation

import numpy as np
import pandas as pd

df = pd.read_csv('Social_Network_Ads.csv')  
x = df[['Age','EstimatedSalary']]           # Indepnedent Feature  
y = df['Purchased']                         # Dependent Feature

from sklearn.model_selection import train_test_split

# K Nearest neighbour Algorithm - it uses equlidian distance concept - it helps to find distance
# Between two dimentional points
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state =5)
# Here we have taken random state = 5 which means it will pick any random data from
# x and assign to x_train and x_test

knnclassifier = KNeighborsClassifier(n_neighbors=5)
knnclassifier.fit(x_train,y_train)        # Fit the model  
y_pred = knnclassifier.predict(x_test)    # Predict the value
metrics.accuracy_score(y_test,y_pred)     # Find Accuracy Scores

x_train.head()


# Now we run the same code again by changing random_state value and we can see a change in the accuracy score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state =9)
knnclassifier = KNeighborsClassifier(n_neighbors=5)
knnclassifier.fit(x_train,y_train)
y_pred = knnclassifier.predict(x_test)
metrics.accuracy_score(y_test,y_pred)

x_train.head()

""" Everytime it will be difficult to change random state value and predict the accuracy.
To build robust model we have cross validation.
It is a tech which involves reserving particular samples of dataset on which you do not 
train the model."""

from sklearn.model_selection import cross_val_score
knnclassifier = KNeighborsClassifier(n_neighbors=4)
print(cross_val_score(knnclassifier, x, y, cv=10, scoring ='accuracy').mean())

""" CV is cross validation values Here we have taken it as 10 means datawill get divided
 in 10 different folds ie 10 differnt experiments and calculate the accuracy
 If we run  - (cross_val_score(knnclassifier, x, y, cv=10, scoring ='accuracy') then 
 you will see all 10 diff accuracy scores.
 
 In the above we have applied Knearest neighbour. We can do it by other algorithm
 lets take Logistic Regression and check the accuracy and decide which Algorithm to use"""

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print (cross_val_score(logreg, x, y, cv=10, scoring = 'accuracy').mean())


