import numpy as np
import pandas as pd
import math
import scipy.stats
from scipy.special import expit

import matplotlib
import matplotlib.pyplot as plt
from pylab import *
%matplotlib inline


# load training data as numpy array
X_train = np.genfromtxt('/Users/Amiros/GitHub/Machine Learning for Data Science/hw2-data/X_train.csv', delimiter=',')
y_train = np.genfromtxt('/Users/Amiros/GitHub/Machine Learning for Data Science/hw2-data/y_train.csv', delimiter=',')

# load test data as numpy array
X_test = np.genfromtxt('/Users/Amiros/GitHub/Machine Learning for Data Science/hw2-data/X_test.csv', delimiter=',')
y_test = np.genfromtxt('/Users/Amiros/GitHub/Machine Learning for Data Science/hw2-data/y_test.csv', delimiter=',')


### 2-A ###
###########

# class index
index_0 = np.where(y_train == 0)[0]
index_1 = np.where(y_train == 1)[0]

# lengths
n = X_train.shape[0]
n_0 = index_0.shape[0]
n_1 = index_1.shape[0]

# Calculate parameters for p(y), theta_Bern, and theta_Pareto
pi_y_0 = index_0.shape[0]/n
pi_y_1 = index_1.shape[0]/n

theta_X_0_Bern = np.mean(X_train[index_0, 0:54], axis=0)
theta_X_1_Bern = np.mean(X_train[index_1, 0:54], axis=0)

theta_X_0_Pareto = n_0/np.log(X_train[index_0, 54:57]).sum(axis=0)
theta_X_1_Pareto = n_1/np.log(X_train[index_1, 54:57]).sum(axis=0)

# Building Bayes classfier
classifier = []

for i in range(X_test.shape[0]):
    Bern_0 = np.multiply(np.power(theta_X_0_Bern, X_test[i, 0:54]), np.power(1 - theta_X_0_Bern, 1-X_test[i, 0:54]))
    Bern_1 = np.multiply(np.power(theta_X_1_Bern, X_test[i, 0:54]), np.power(1 - theta_X_1_Bern, 1-X_test[i, 0:54]))

    Prato_0 = np.multiply(theta_X_0_Pareto, np.power(X_test[i, 54:57], -1 - theta_X_0_Pareto))
    Prato_1 = np.multiply(theta_X_1_Pareto, np.power(X_test[i, 54:57], -1 - theta_X_1_Pareto))


    full_row_0 = np.concatenate((Bern_0, Prato_0), axis=0)
    full_row_1 = np.concatenate((Bern_1, Prato_1), axis=0)

    p = [np.prod(full_row_0) * pi_y_0, np.prod(full_row_1) * pi_y_1]  
    classifier.append(p)

classifier_df = pd.DataFrame(classifier, columns=[0, 1])


# Find the column name which has the maximum value for each row
y_predict = classifier_df.idxmax(axis=1).tolist()

# Confusion Matrix
(y_predict == y_test).sum()

y_actu = pd.Series(y_test, name='Actual')
y_predict = pd.Series(y_predict, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_predict)

accu = (54 + 32) / 93
print(df_confusion, '\n', '\n', "Accuracy:", accu)


### 2-B ###
###########

plt.figure(figsize=(12, 8))

plt.stem(theta_X_1_Bern)
plt.setp(plt.stem(theta_X_0_Bern), linewidth=.75, color='r')
plt.xlim(-.05, 56)
plt.ylim(0, 1)
plt.title('Stem plot for Bernoulli parameters of each class')
labels = ['Class 0', 'Class 1']
plt.legend(labels)

plt.show()

### 2-C ###
###########

# Function to implement K Nearest Neighbours
def knn_classifier(training_data, train_label, test_data, test_label, k): 
    dist = np.empty([training_data.shape[0], ])
    knn_prediction = pd.DataFrame()
 
    for i in range(test_data.shape[0]):
        dist = np.column_stack((dist, np.sum(np.abs(test_data[i, :] - training_data), axis=1)))

    for i in range(test_data.shape[0]):
        for k in range(1, k+1):
            knn_index = dist[:, i].argsort()[:k]
            training_lables = y_train[knn_index]

            knn_prediction.set_value(i, k, scipy.stats.mode(training_lables)[0][0]) 
 
    # accuracy
    knn_arr = knn_prediction.values
    knn_accuracy = []

    for i in range(0, k):
        correct_predict = np.where(knn_arr[:, i] == test_label)[0].shape[0]
        knn_accuracy.append(correct_predict/93)

    return knn_prediction, knn_accuracy


# Calling knn_classifier function
knn_prediction, acc = knn_classifier(X_train, y_train, X_test, y_test, 20)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(acc)
plt.xlim([0, 20])
plt.xticks(np.arange(0, 20, 1))

# Chart title
plt.title('kNN prediction accuracy as a function of k')


### 2-D ###
###########
# replacing all 0s with -1
y_train_reg, y_test_reg = y_train, y_test
y_train_reg = y_train_reg.reshape(-1, 1)

y_train_reg[y_train_reg == 0] = -1
y_test_reg[y_test_reg == 0] = -1



# add an extra dimesnion to data
X_train_reg = np.concatenate((X_train, np.tile(1, (X_train.shape[0], 1))), axis=1)
X_test_reg = np.concatenate((X_test, np.tile(1, (X_test.shape[0], 1))), axis=1)

def sigmoid(scores):
    return expit(scores)

# Initializing the parameters
num_iteration = 10000
weights = np.zeros((1,X_train_reg.shape[1]))
obj_function = np.empty([0,0])

for steps in range(1, num_iteration + 1):  
    # setting the learning rate
    learning_rate = 1e-05/np.sqrt(1 + steps)

      
    scores = np.dot(np.multiply(y_train_reg, X_train_reg), weights.T)

    s = sigmoid(scores)
    # Updated weights with gradient ascend
    output_error = np.multiply((1-s), y_train_reg)

    gradient = np.dot(X_train_reg.T, output_error)
    
    weights += learning_rate * gradient.T

    # Appending the objective value of iteratio
    obj_function = np.append(obj_function, np.sum(np.log(s)))

plt.figure(figsize=(12, 8))

x = range(0, 10000)
plot(obj_function)

plt.xlim(0, 10000)
plt.title('Objective function value for each iteration')
plt.xlabel('iteration')
plt.ylabel('objective function value')

### 2-E ###
###########
num_iteration = 100
weights = np.zeros((1,X_train_reg.shape[1]))
obj_function = np.empty([0,0])

for steps in range(1,num_iteration + 1):   
    # setting the learning rate
    learning_rate = 1/np.sqrt( 1 + steps)
    
    scores = np.dot(np.multiply(y_train_reg, X_train_reg), weights.T)
    
    # Gradient
    s = sigmoid(scores)
    output_error = np.multiply((1-s), y_train_reg)
    gradient = np.dot(X_train_reg.T, output_error)
    
    # Hessian
    s_h = s * (1-s)
    hessian = np.dot(X_train_reg.T, np.multiply(ss, X_train_reg))
    hessian_invers = np.linalg.inv(gg)
    
    # adding Gradient and Hessian
    t = np.dot(hessian_invers, gradient)
    
    weights += learning_rate * t.T

    # Appending the objective value of iteratio
    obj_function = np.append(obj_function, np.sum(np.log(s)))


plt.figure(figsize=(12,8))

x = range(0,10000)
plot(obj_function)

plt.xlim(0,100)
plt.title('Objective function value for each iteration')
plt.xlabel('iteration')
plt.ylabel('objective function value')

# calculating accuracy of prediction
e_predict = sigmoid(np.dot(X_test_reg, weights.T))

e_predict[e_predict >= 0.5] = 1
e_predict[e_predict < 0.5] = -1

e_accuracy = (e_predict == y_test_reg).sum()/y_test_reg.shape[0]

figtext(.95, .9, "Accuracy: 0.91397849462365588")
