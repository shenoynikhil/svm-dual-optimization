#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import csv
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
# libsvm implementation
from sklearn.svm import SVC
from functions import get_error_libsvm, accuracy_score, perform_cvx_opt, train_test_split

#%% Our main dataset
data = []
with open('./data.csv','r')as f:
  csv_reader = csv.reader(f)
  for line in csv_reader:
      data.append([float(i) for i in line])

X = np.array([i[:-1] for i in data]).astype(float)
y = np.array([i[-1] for i in data]).astype(int)
y = np.reshape(y,(len(y), 1))

#%% Use CVX to obtain results for a linear kernel
# Set target variables as 1 or -1
a = 4
b = 7
# from SVC results
c_val = 0.075
X_sub = np.array([X[j] for j in [i for i, x in enumerate(y) if x == a or x == b]])
y_sub = np.array([i for i in y if i == a or i == b])

X_cvx = X_sub
y_cvx = y_sub
y_cvx[y_cvx == a] = 1
y_cvx[y_cvx == b] = -1

X_cvx_train, y_cvx_train, X_cvx_test, y_cvx_test = train_test_split(X_cvx, y_cvx, split = 0.75)

w, intercept, alphas = perform_cvx_opt(X_cvx_train, y_cvx_train, kernel = 'linear', C = c_val)

#%% Do predictions 
y_pred = np.matmul(X_cvx_test, w) + intercept
y_pred = np.array([int(i[0]/abs(i[0])) for i in y_pred])

print ("Accuracy using cvx for binary classification : " + str(accuracy_score(y_cvx_test, y_pred)))

#%% Polynomial Kernel
w, intercept, a = perform_cvx_opt(X_cvx_train, y_cvx_train, kernel = 'poly', C = 0.005157, 
                                  gamma = 0.07878, p = 5)

#%% Do predictions 
y_pred = np.matmul(X_cvx_test, w) + intercept
y_pred = np.array([int(i[0]/abs(i[0])) for i in y_pred])

print ("Accuracy using cvx for binary classification : " + str(accuracy_score(y_cvx_test, y_pred)))
