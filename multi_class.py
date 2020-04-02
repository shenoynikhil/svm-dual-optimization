#%% import libraries
import csv
import numpy as np
# libsvm implementation
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
from functions import train_test_split, cross_validation_split, accuracy_score, get_error_libsvm

#%% Our main dataset
data = []
with open('./data.csv','r')as f:
  csv_reader = csv.reader(f)
  for line in csv_reader:
      data.append([float(i) for i in line])

X = np.array([i[:-1] for i in data]).astype(float)
#X = X[:, :10]
y = np.array([i[-1] for i in data]).astype(int)
y = np.reshape(y,(len(y), 1))

############################################## Linear Kernel ###################################################

#%% Linear Kernel : Findina Optimal Values of Hyperparameters using 2D Grid Search
C_values = np.linspace(1e-2, 1, 30)
k_fold = 10

acc_storage = np.empty(shape = (len(C_values), 1))
for i in range(len(C_values)):
    c_val = C_values[i]
    
    acc_storage[i] = get_error_libsvm(X, y, kernel = 'linear', c_val = c_val, 
               k_fold = k_fold, decision_function_shape_val='ovr')

#%% Results from Grid Search 
print("Maximum accuracy " + str(np.amax(acc_storage)))
index = np.where(acc_storage == np.amax(acc_storage))
print("C_value for max accuracy : " + str(C_values[index[0][0]]))

#%% Get Results on train test Data for linear kernel
X_train, y_train, X_test, y_test = train_test_split(X, y, split = 0.75)

c_val = C_values[index[0][0]]

clf = SVC(C = c_val, kernel = 'linear', decision_function_shape = 'ovr')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print ("Training Error is : " + str(100 * accuracy_score(clf.predict(X_train), y_train)) + "%")
print ("Testing Error is : " + str(100 * accuracy_score(y_pred, y_test)) + "%")

############################################## Gaussian Kernel ###################################################

#%% RBF Kernel : Findina Optimal Values of Hyperparameters using 2D Grid Search
C_values = np.linspace(0.07, 0.8, 20)
gamma_values = np.linspace(0.05, 0.5, 20)
k_fold = 10

acc_storage = np.empty(shape = (len(C_values), len(gamma_values)))
for i in range(len(C_values)):
    for j in range(len(gamma_values)):
        c_val = C_values[i]
        gamma_val = gamma_values[j]
        
        acc_storage[i][j] = get_error_libsvm(X, y, 'rbf', c_val, gamma_val = gamma_val,
                   k_fold = k_fold, decision_function_shape_val = 'ovr')

#%% Results from 2D Grid Search 
print("Maximum accuracy " + str(np.amax(acc_storage)))
index = np.where(acc_storage == np.amax(acc_storage))
print("C_value for max accuracy : " + str(C_values[index[0][0]]))
print("Gamma Value for Parameter for best fit : " + str(gamma_values[index[1][0]]))

#%% Get Results for train test data for rbf kernel
X_train, y_train, X_test, y_test = train_test_split(X, y, split = 0.75)

#c_val = C_values[index[0][0]]
#gamma_val = gamma_values[index[1][0]]
c_val = 0.5
gamma_val = 0.099

clf = SVC(C = c_val, kernel = 'rbf', gamma=gamma_val, decision_function_shape='ovr')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print ("Training Error is : " + str(100 * accuracy_score(clf.predict(X_train), y_train)) + "%")
print ("Testing Error is : " + str(100 * accuracy_score(y_pred, y_test)) + "%")

############################################## Polynomial Kernel ###################################################
#%% Polynomial Kernel : Findina Optimal Values of Hyperparameters using 2D Grid Search
C_values = np.linspace(0.1, 1.3, 25)
gamma_values = np.linspace(1e-2, 1, 25)
p_values = np.array(range(2, 8))
k_fold = 10

acc_storage = np.empty(shape = (len(C_values), len(gamma_values), len(p_values)))
for i in range(len(C_values)):
    for j in range(len(gamma_values)):
        for k in range(len(p_values)):
            c_val = C_values[i]
            gamma_val = gamma_values[j]
            p_val = p_values[k]
            
            acc_storage[i][j][k] = get_error_libsvm(X, y, 'poly', c_val,
                       gamma_val = gamma_val, k_fold = k_fold, p_val = p_val, decision_function_shape_val='ovr')
            

#%% Results from 2D Grid Search 
print("Maximum accuracy " + str(np.amax(acc_storage)))
index = np.where(acc_storage == np.amax(acc_storage))
print("C_value for max accuracy : " + str(C_values[index[0][0]]))
print("Gamma Value of Parameter for best fit : " + str(gamma_values[index[1][0]]))
print("Degree Value of Parameter for best fit : " + str(p_values[index[2][0]]))

#%% Get Results for train test data for rbf kernel
X_train, y_train, X_test, y_test = train_test_split(X, y, split = 0.75)

c_val = C_values[index[0][0]]
gamma_val = gamma_values[index[1][0]]
degree_val = p_values[index[2][0]]

clf = SVC(C = c_val, kernel = 'poly', gamma=gamma_val, degree = degree_val, decision_function_shape='ovr')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print ("Training Accuracy is : " + str(100 * accuracy_score(clf.predict(X_train), y_train)) + "%")
print ("Testing Accuracy is : " + str(100 * accuracy_score(y_pred, y_test)) + "%")



