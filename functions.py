
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:06:18 2019

@author: Nikhil
"""
#%% Import Libraries
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

#%%
# Split a dataset into a train and test set
def train_test_split(X, y, split = 0.60):
    n = int(split * len(X))
    test_size = int((1 - split) * len(X))
    X_train = X[0:n,]
    y_train = np.reshape(y[0:n,], (n, 1))
    
    X_test = X[n:,]
    y_test = np.reshape(y[n:,], (test_size, 1))
    
    return X_train, y_train, X_test, y_test

# Split a dataset into k folds
def cross_validation_split(X, y, folds):
    X_split = list()
    y_split = list()
    X_copy = list(X)
    y_copy = list(y)
    fold_size = int(len(X) / folds)
    # Create splits now :
    for i in range(folds):
        X_fold = list()
        y_fold = list()
        while len(X_fold) < fold_size:
            index = randrange(len(X_copy))
            X_fold.append(X_copy.pop(index))
            y_fold.append(y_copy.pop(index))
        X_split.append(np.array(X_fold))
        y_split.append(np.array(y_fold))
    return X_split, y_split

# Define kernels
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, gamma=1e-2):
    return np.exp(-gamma * np.linalg.norm(x-y)**2)

def perform_cvx_opt(X, y, C = None, soft_threshold = 1e-4, kernel = 'linear', p = 3, gamma = 1e-1):
    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if kernel == 'linear':
                K[i,j] = linear_kernel(X[i], X[j])
            elif kernel == 'poly':
                K[i,j] = polynomial_kernel(X[i], X[j], p)
            elif kernel == 'rbf' :
                K[i, j] = gaussian_kernel(X[i], X[j], gamma)
                
    P = cvxopt_matrix(np.outer(y, y) * K)
    q = cvxopt_matrix(np.ones(n_samples) * -1)
    A = cvxopt_matrix(y.reshape(1, -1))
    A = cvxopt_matrix(A, (1, n_samples), 'd')
    b = cvxopt_matrix(0.0)
    
    if C is None:
        G = cvxopt_matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt_matrix(np.zeros(n_samples))
    else:
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt_matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * C
        h = cvxopt_matrix(np.hstack((tmp1, tmp2)))
    
    # solve QP problem
    solution = cvxopt_solvers.qp(P, q, G, h, A, b)
    
    # Lagrange multipliers
    a = np.ravel(solution['x'])

    # Calculate weights
    w = np.matrix(np.zeros((1, n_features)))
    for i in range(n_samples):
        w += a[i] * y[i] * X[i]
        
    # Calculate Intercepts
    intercept = 0
    for i in range(n_samples):
        if a[i] > soft_threshold:
            intercept = y[i] - w.dot(np.transpose(X[i]))
            break
    
    intercept = float(intercept)
    
    return w.T, intercept, a

def accuracy_score(y_true, y_pred):
    count = 0.0
    for i in range(len(y_true)):
        if(y_true[i] == y_pred[i]):
            count += 1
    
    return count/float(len(y_true))

def get_error_libsvm(X, y, kernel, c_val = 0, gamma_val = 'auto',
                     k_fold = 4, p_val = 3, decision_function_shape_val = 'ovo'):
    X_split, y_split = cross_validation_split(X, y, folds = k_fold)
    total_error = 0
    for i in range(k_fold):
        X_test_k_fold = X_split[i]
        y_test_k_fold = y_split[i]
        X_train_k_fold = np.empty(shape = (0, X_test_k_fold.shape[1]))
        y_train_k_fold = np.empty(shape = (0, 1))
        for j in range(k_fold):
            if(j != i):
                X_train_k_fold = np.concatenate((X_train_k_fold, X_split[i]), axis = 0)
                y_train_k_fold = np.concatenate((y_train_k_fold, y_split[i]), axis = 0)
        
        clf = SVC(C = c_val, kernel=kernel, gamma=gamma_val,
                  degree = p_val, decision_function_shape = decision_function_shape_val) 
        clf.fit(X_train_k_fold, y_train_k_fold)
        
        y_pred = clf.predict(X_test_k_fold)
        temp_accuracy = accuracy_score(y_pred, y_test_k_fold)
        
        total_error += temp_accuracy
    
    return total_error/k_fold


