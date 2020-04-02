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

#%% Create binary class 
a = 5
b = 9
X_sub = np.array([X[j] for j in [i for i, x in enumerate(y) if x == a or x == b]])
y_sub = np.array([i for i in y if i == a or i == b])

############################################## Linear Kernel ###################################################

#%% Linear Kernel : Findina Optimal Values of Hyperparameters using 2D Grid Search
C_values = np.linspace(1e-3, 0.08, 100)
k_fold = 6

acc_storage = np.empty(shape = (len(C_values), 1))
for i in range(len(C_values)):
    c_val = C_values[i]
    
    acc_storage[i] = get_error_libsvm(X_sub, y_sub, kernel = 'linear', c_val = c_val, k_fold = 6)

#%% Results from Grid Search 
print("Maximum accuracy " + str(np.amax(acc_storage)))
index = np.where(acc_storage == np.amax(acc_storage))
print("C_value for max accuracy : " + str(C_values[index[0][0]]))

#%% Get Results on train test Data for linear kernel
X_sub_train, y_sub_train, X_sub_test, y_sub_test = train_test_split(X_sub, y_sub, split = 0.75)

c_val = C_values[index[0][0]]

clf = SVC(C = c_val, kernel = 'linear')
clf.fit(X_sub_train, y_sub_train)

y_pred = clf.predict(X_sub_test)
print (accuracy_score(y_pred, y_sub_test))

#%% Linear Kernel
C_values = np.linspace(1e-3, 0.08, 100)
k_fold = 10
X_split, y_split = cross_validation_split(X_sub, y_sub, folds = k_fold)
cross_val_scores = []

for c_val in C_values:
    total_acc = 0
    for i in range(k_fold):
        X_test_k_fold = X_split[i]
        y_test_k_fold = y_split[i]
        X_train_k_fold = np.empty(shape = (0, X_test_k_fold.shape[1]))
        y_train_k_fold = np.empty(shape = (0, 1))
        for j in range(k_fold):
            if(j != i):
                X_train_k_fold = np.concatenate((X_train_k_fold, X_split[i]), axis = 0)
                y_train_k_fold = np.concatenate((y_train_k_fold, y_split[i]), axis = 0)
        
        clf = SVC(C = c_val, kernel='linear') 
        clf.fit(X_train_k_fold, y_train_k_fold)
        
        y_pred = clf.predict(X_test_k_fold)
        temp_accuracy = accuracy_score(y_pred, y_test_k_fold)
        
        total_acc += temp_accuracy
    cross_val_scores.append(1 - total_acc/k_fold)
    
#%% Plot variation of C and cross validation errors
plt.plot(C_values, cross_val_scores)
plt.show()

#%% Variation of C using train-test split using a linear kernel
C_values = np.linspace(1e-4, 0.35, 50)
test_accuracy = []
train_accuracy = []
X_sub_train, y_sub_train, X_sub_test, y_sub_test = train_test_split(X_sub, y_sub, split = 0.75)
for c_val in C_values:
    clf = SVC(C = c_val, kernel='linear') 
    clf.fit(X_sub_train, y_sub_train)
    
    y_test_pred = clf.predict(X_sub_test)
    test_accuracy.append(1 - accuracy_score(y_test_pred, y_sub_test))
    
    y_train_pred= clf.predict(X_sub_train)
    train_accuracy.append(1 - accuracy_score(y_train_pred, y_sub_train))
   
#%% Plot variation of C and train-test errors
plt.plot(C_values, test_accuracy, 'r')
plt.plot(C_values, train_accuracy, 'b')
plt.show()

############################################## Gaussian Kernel ###################################################

#%% RBF Kernel : Findina Optimal Values of Hyperparameters using 2D Grid Search
C_values = np.linspace(1e-3, 0.08, 20)
gamma_values = np.linspace(1e-3, 1e-1, 15)
k_fold = 6

acc_storage = np.empty(shape = (len(C_values), len(gamma_values)))
for i in range(len(C_values)):
    for j in range(len(gamma_values)):
        c_val = C_values[i]
        gamma_val = gamma_values[j]
        
        acc_storage[i][j] = get_error_libsvm(X_sub, y_sub, 'rbf', c_val,  gamma_val = gamma_val, k_fold = k_fold)

#%% Results from 2D Grid Search 
print("Maximum accuracy " + str(np.amax(acc_storage)))
index = np.where(acc_storage == np.amax(acc_storage))
print("C_value for max accuracy : " + str(C_values[index[0][0]]))
print("Gamma Value for Parameter for best fit : " + str(gamma_values[index[1][0]]))

#%% Get Results for train test data for rbf kernel
X_sub_train, y_sub_train, X_sub_test, y_sub_test = train_test_split(X_sub, y_sub, split = 0.75)

c_val = C_values[index[0][0]]
gamma_val = gamma_values[index[1][0]]

clf = SVC(C = c_val, kernel = 'rbf', gamma=gamma_val)
clf.fit(X_sub_train, y_sub_train)

y_pred = clf.predict(X_sub_test)
print (accuracy_score(y_pred, y_sub_test))


#%% Variation of C using train-test split using a linear kernel
C_values = np.linspace(0.020, 0.05, 50)
gamma_val = gamma_values[index[1][0]]
test_accuracy = []
train_accuracy = []

X_sub_train, y_sub_train, X_sub_test, y_sub_test = train_test_split(X_sub, y_sub, split = 0.75)
for c_val in C_values:
    clf = SVC(C = c_val, kernel='rbf', gamma = gamma_val) 
    clf.fit(X_sub_train, y_sub_train)
    
    y_test_pred = clf.predict(X_sub_test)
    test_accuracy.append(1 - accuracy_score(y_test_pred, y_sub_test))
    
    y_train_pred= clf.predict(X_sub_train)
    train_accuracy.append(1 - accuracy_score(y_train_pred, y_sub_train))
   
#%% Plot variation of C and train-test errors
plt.plot(C_values, test_accuracy, 'r')
plt.plot(C_values, train_accuracy, 'b')
plt.show()

#%% Variation of Gamma using train-test split using a linear kernel
c_val = C_values[index[0][0]]
gamma_values = np.linspace(1e-3, 1e-1, 15)
test_accuracy = []
train_accuracy = []

X_sub_train, y_sub_train, X_sub_test, y_sub_test = train_test_split(X_sub, y_sub, split = 0.75)
for gamma_val in gamma_values:
    clf = SVC(C = c_val, kernel='rbf', gamma = gamma_val) 
    clf.fit(X_sub_train, y_sub_train)
    
    y_test_pred = clf.predict(X_sub_test)
    test_accuracy.append(1 - accuracy_score(y_test_pred, y_sub_test))
    
    y_train_pred= clf.predict(X_sub_train)
    train_accuracy.append(1 - accuracy_score(y_train_pred, y_sub_train))
   
#%% Plot variation of C and train-test errors
plt.plot(gamma_values, test_accuracy, 'r')
plt.plot(gamma_values, train_accuracy, 'b')
plt.show()

############################################## Polynomial Kernel ###################################################

#%% Polynomial Kernel : Findina Optimal Values of Hyperparameters using 2D Grid Search
C_values = np.linspace(1e-3, 0.08, 20)
gamma_values = np.linspace(1e-3, 1e-1, 15)
p_values = np.array(range(2, 8))
k_fold = 6

acc_storage = np.empty(shape = (len(C_values), len(gamma_values), len(p_values)))
for i in range(len(C_values)):
    for j in range(len(gamma_values)):
        for k in range(len(p_values)):
            c_val = C_values[i]
            gamma_val = gamma_values[j]
            p_val = p_values[k]
            
            acc_storage[i][j][k] = get_error_libsvm(X_sub, y_sub, 'poly', c_val,
                       gamma_val = gamma_val, k_fold = k_fold, p_val = p_val)
            

#%% Results from 2D Grid Search 
print("Maximum accuracy " + str(np.amax(acc_storage)))
index = np.where(acc_storage == np.amax(acc_storage))
print("C_value for max accuracy : " + str(C_values[index[0][0]]))
print("Gamma Value of Parameter for best fit : " + str(gamma_values[index[1][0]]))
print("Degree Value of Parameter for best fit : " + str(p_values[index[2][0]]))


#%% Get Results for train test data for rbf kernel
X_sub_train, y_sub_train, X_sub_test, y_sub_test = train_test_split(X_sub, y_sub, split = 0.75)

c_val = C_values[index[0][0]]
gamma_val = gamma_values[index[1][0]]
degree_val = p_values[index[2][0]]

clf = SVC(C = c_val, kernel = 'poly', gamma=gamma_val, degree = degree_val)
clf.fit(X_sub_train, y_sub_train)

y_pred = clf.predict(X_sub_test)
print (accuracy_score(y_pred, y_sub_test))

#%% Variation of C using train-test split using a linear kernel
C_values = np.linspace(1e-3, 0.02, 100)
gamma_val = gamma_values[index[1][0]]
degree_val = p_values[index[2][0]]
test_accuracy = []
train_accuracy = []

X_sub_train, y_sub_train, X_sub_test, y_sub_test = train_test_split(X_sub, y_sub, split = 0.75)
for c_val in C_values:
    clf = SVC(C = c_val, kernel='poly', gamma = gamma_val, degree = degree_val) 
    clf.fit(X_sub_train, y_sub_train)
    
    y_test_pred = clf.predict(X_sub_test)
    test_accuracy.append(1 - accuracy_score(y_test_pred, y_sub_test))
    
    y_train_pred= clf.predict(X_sub_train)
    train_accuracy.append(1 - accuracy_score(y_train_pred, y_sub_train))
   
#%% Plot variation of C and train-test errors
plt.plot(C_values, test_accuracy, 'r')
plt.plot(C_values, train_accuracy, 'b')
plt.show()

#%% Variation of Gamma using train-test split using a linear kernel
c_val = C_values[index[0][0]]
gamma_values = np.linspace(1e-3, 1e-1, 50)
degree_val = p_values[index[2][0]]
test_accuracy = []
train_accuracy = []

X_sub_train, y_sub_train, X_sub_test, y_sub_test = train_test_split(X_sub, y_sub, split = 0.75)
for gamma_val in gamma_values:
    clf = SVC(C = c_val, kernel='poly', gamma = gamma_val, degree = degree_val) 
    clf.fit(X_sub_train, y_sub_train)
    
    y_test_pred = clf.predict(X_sub_test)
    test_accuracy.append(1 - accuracy_score(y_test_pred, y_sub_test))
    
    y_train_pred= clf.predict(X_sub_train)
    train_accuracy.append(1 - accuracy_score(y_train_pred, y_sub_train))
   
#%% Plot variation of C and train-test errors
plt.plot(gamma_values, test_accuracy, 'r')
plt.plot(gamma_values, train_accuracy, 'b')
plt.show()


