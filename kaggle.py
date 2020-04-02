#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:09:14 2019

@author: Nikhil
"""

import csv
import numpy as np
import pandas as pd
# libsvm implementation
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
from functions import train_test_split, cross_validation_split, accuracy_score, get_error_libsvm

#%% Our main dataset
data = []
with open('./kaggle_train.csv','r')as f:
  csv_reader = csv.reader(f)
  for line in csv_reader:
      data.append([float(i) for i in line])

X = np.array([i[:-1] for i in data]).astype(float)
#X = X[:, :10]
y = np.array([i[-1] for i in data]).astype(int)
y = np.reshape(y,(len(y), 1))

data = []
with open('./kaggle_test.csv','r')as f:
  csv_reader = csv.reader(f)
  for line in csv_reader:
      data.append([float(i) for i in line])

X_kaggle = np.array([i for i in data]).astype(float)


#%% RBF Kernel : Findina Optimal Values of Hyperparameters using 2D Grid Search
C_values = np.linspace(0.2, 1.1, 5)
t = 1 / (25 * X.var())
gamma_val = t
k_fold = 10

acc_storage = np.empty(shape = (len(C_values)))
for i in range(len(C_values)):
    c_val = C_values[i]
    gamma_val = gamma_val
    
    acc_storage[i] = get_error_libsvm(X, y, 'rbf', c_val, gamma_val = gamma_val,
               k_fold = k_fold, decision_function_shape_val = 'ovo')

#%% Results from 2D Grid Search 
print("Maximum accuracy " + str(np.amax(acc_storage)))
index = np.where(acc_storage == np.amax(acc_storage))
print("C_value for max accuracy : " + str(C_values[index[0][0]]))
#print("Gamma Value for Parameter for best fit : " + str(gamma_values[index[1][0]]))

#%% Get Results for train test data for rbf kernel
X_train, y_train, X_test, y_test = train_test_split(X, y, split = 0.75)

#c_val = C_values[index[0][0]]
#gamma_val = gamma_values[index[1][0]]
c_val = 0.8
gamma_val = 1 / (25 * X.var())

clf = SVC(kernel = 'poly', gamma=gamma_val, decision_function_shape='ovr', C = c_val)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print ("Training Accuracy is : " + str(100 * accuracy_score(clf.predict(X_train), y_train)) + "%")
print ("Testing Accuracy is : " + str(100 * accuracy_score(y_pred, y_test)) + "%")

#%%
y_kaggle = clf.predict(X_kaggle)
df = pd.DataFrame(y_kaggle)
df.to_csv('test.csv')

#%% 
C_values = np.linspace(0.02, 1.5, 20)
test_accuracy = []
train_accuracy = []

X_train, y_train, X_test, y_test = train_test_split(X, y, split = 0.75)
for c_val in C_values:
    clf = SVC(C = c_val, kernel='rbf', gamma = 'scale') 
    clf.fit(X_train, y_train)
    
    y_test_pred = clf.predict(X_test)
    test_accuracy.append(accuracy_score(y_test_pred, y_test))
    
    y_train_pred= clf.predict(X_train)
    train_accuracy.append(accuracy_score(y_train_pred, y_train))

#%% Plot variation of C and train-test errors
plt.plot(C_values, test_accuracy, 'r')
plt.plot(C_values, train_accuracy, 'b')
plt.show()