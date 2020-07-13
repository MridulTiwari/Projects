# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 23:31:14 2020

@author: mridul
"""

import pandas as pd ##Data manipulation and data analysis
import numpy as np ##Support for large multi-dimensional arrays and matrix
import seaborn as sb ## Statistical plotting of data like styles,color
import matplotlib.pyplot as plt ## For plotting

##sklearn-all data-mining concepts which are interoperate with python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split ##train and test split
from sklearn import metrics ## accuracy calculation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


##Loading the data
Bank_loan = pd.read_csv("D:\\EDWISOR\\Project Loan default\\Bank_loan.csv")
print(Bank_loan.head(5))

##Getting the barplot for the categorical columns
sb.countplot(x="ed",data=Bank_loan,palette="hls")
Bank_loan.ed.value_counts() ##For numerical count

sb.countplot(x="default",data=Bank_loan,palette="hls")
Bank_loan.default.value_counts() ##For numerical count

# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns

## x= ed
sb.boxplot(x="ed",y="age",data=Bank_loan,palette="hls")

sb.boxplot(x="ed",y="employ",data=Bank_loan,palette="hls")

sb.boxplot(x="ed",y="address",data=Bank_loan,palette="hls")

sb.boxplot(x="ed",y="income",data=Bank_loan,palette="hls")

sb.boxplot(x="ed",y="debtinc",data=Bank_loan,palette="hls")

sb.boxplot(x="ed",y="creddebt",data=Bank_loan,palette="hls")

sb.boxplot(x="ed",y="othdebt",data=Bank_loan,palette="hls")

## x= default

sb.boxplot(x="default",y="age",data=Bank_loan,palette="hls")

sb.boxplot(x="default",y="employ",data=Bank_loan,palette="hls")

sb.boxplot(x="default",y="address",data=Bank_loan,palette="hls")

sb.boxplot(x="default",y="income",data=Bank_loan,palette="hls")

sb.boxplot(x="default",y="debtinc",data=Bank_loan,palette="hls")

sb.boxplot(x="default",y="creddebt",data=Bank_loan,palette="hls")

sb.boxplot(x="default",y="othdebt",data=Bank_loan,palette="hls")


# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns



##convert the types
Bank_loan['default'] = pd.Categorical(Bank_loan.default)
print (Bank_loan.dtypes)
Bank_loan["ed"] = pd.Categorical(Bank_loan.ed)
print(Bank_loan.dtypes)

##count the na value
Bank_loan.isnull().sum()

##Fill nan values with mode of categorical coloumn
##Mode value imputation
Bank_loan.default.mode()
Bank_loan["default"].fillna(0,inplace=True) #mode of default variable is 0

##Check again the na value
Bank_loan.isnull().sum()


##Model Building (Define X and Y)

X = Bank_loan.iloc[:,[0,1,2,3,4,5,6,7]]
Y = Bank_loan.iloc[:,8]

##Split the data

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.3, random_state=1)

classifier = LogisticRegression()
classifier.fit(X,Y)

print (classifier.intercept_,classifier.coef_) # coeficient of features
prob = classifier.predict_proba (X_test) ##Probality values

##Accuracy on train data 
predict_train = classifier.predict(X_train)
print('Target on train data',predict_train)

accuracy_train = accuracy_score(Y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_train,predict_train)
print(confusion_matrix)




##Accuracy on test data
predict_test = classifier.predict(X_test)
print('Target on test data',predict_test) 

accuracy_test =accuracy_score(Y_test,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test,predict_test)
print(confusion_matrix)




y_prob = pd.DataFrame(classifier.predict_proba(X_train.iloc[:,:]))
fig,ax = plt.subplots()
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive");
##ROC curve
fpr,tpr,thresholds = metrics.roc_curve(Y_train,y_prob.iloc[:,1:])




roc_auc = metrics.auc(fpr, tpr)
roc_auc  ##0.8441464821222607


##Decision trees
import pandas as pd
import matplotlib.pyplot as plt
Bank_loan = pd.read_csv("D:\\EDWISOR\\Project Loan default\\Bank_loan.csv")
Bank_loan.head() #It shows the top 5 observation
Bank_loan["default"].unique() ##It show the number of cateogry in default variable
Bank_loan.default.value_counts()##It shows the nuber of zero and number  of 1 present in default variable
##count the na value
Bank_loan.isnull().sum()

##Fill nan values with mode of categorical coloumn
##Mode value imputation
Bank_loan.default.mode()
Bank_loan["default"].fillna(0,inplace=True) #mode of default variable is 0

##Check again the na value
Bank_loan.isnull().sum()
colnames = list(Bank_loan.columns)##It make list of all the variables in Bank_loan
predictors = colnames[:8] ##It 
target = colnames[8]

##Splitting the data into train and test dataset
import numpy as np
from sklearn.model_selection import train_test_split
train,test = train_test_split(Bank_loan,test_size = 0.3,random_state = 1)

train.default.value_counts()
test.default.value_counts()

from sklearn.tree import DecisionTreeClassifier


model = DecisionTreeClassifier(criterion = "entropy")
model.fit(train[predictors],train[target])

##Accuracy for train
np.mean(pd.Series(train.default).reset_index(drop=True) == pd.Series(model.predict(train[predictors])))

##Accuracy for test 
np.mean(pd.Series(test.default).reset_index(drop=True) == pd.Series(model.predict(test[predictors])))

















































