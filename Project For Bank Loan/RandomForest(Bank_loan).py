# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 02:29:48 2020

@author: mridul
"""


import pandas as pd ##Data manipulation and data analysis
import numpy as np ##Support for large multi-dimensional arrays and matrix
import seaborn as sb ## Statistical plotting of data like styles,color
import matplotlib.pyplot as plt ## For plotting

##sklearn-all data-mining concepts which are interoperate with python

from sklearn.model_selection import train_test_split ##train and test split
from sklearn import metrics ## accuracy calculation
from sklearn.metrics import classification_report

##Loading the data
Bank_loan = pd.read_csv("D:\\EDWISOR\\Project Loan default\\Bank_loan.csv")
##Fill nan values with mode of categorical coloumn
##Mode value imputation
Bank_loan.default.mode()
Bank_loan["default"].fillna(0,inplace=True) #mode of default variable is 0

##Check again the na value
Bank_loan.isnull().sum()

Bank_loan.head()
Bank_loan["default"].unique()
Bank_loan.default.value_counts()
colnames = list(Bank_loan.columns)


##Splitting the data into train and test dataset

train,test = train_test_split(Bank_loan,test_size = 0.3,random_state = 1)

train.default.value_counts()
test.default.value_counts()


colnames = train.columns
len(colnames[0:8])
trainX = train[colnames[0:8]]
trainY = train[colnames[8]]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=15,criterion="entropy")

rf.fit(trainX,trainY) 


##Accuracy of training data by classifier method
predictions = rf.predict(trainX)
classification_report(trainY,predictions)

# Check the accuracy by confusion matrix

trainX["rf_pred"] = rf.predict(trainX)
from sklearn.metrics import confusion_matrix
confusion_matrix(trainY,trainX["rf_pred"]) # Confusion matrix
print ("Accuracy",(459+132)/(459+133+0+3)) ## 99.32




# Accuracy on testing data 
testX = test[colnames[0:8]]
testY = test[colnames[8]]

##Accuracy of test data by classifier method
rf.fit(testX,testY)
predictions1= rf.predict(testX)
classification_report(testY,predictions1)

##Check the accuracy of test data by confusion matrix

testX["rf_pred"] = rf.predict(testX)
confusion_matrix(testY,testX["rf_pred"])
print ("Accuracy",(208+45)/(208+45+2+0)) # 99.21




















































