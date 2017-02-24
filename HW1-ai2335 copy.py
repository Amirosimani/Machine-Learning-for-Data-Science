import numpy as np
from numpy.linalg import inv
import math

import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


def fit(x_train, y_train, lmda):    
    """ Fit Ridge Regression.

    This function  takes 3 inputs. `x_train`, `y_train`, and `lmda`. 

    Parameters
    ----------
    x_train : numpy-array
        a dataset represented as a numpy-array.
    y_train : numpy-array
        a dataset represented as a numpy-array.
    lmda  : int

    Returns
    -------
    numpy-array
    Wrr: weigth vector for Ridge Regression.
    """

    a = x_train.T.dot(x_train) + lmda*np.eye(x_train.shape[1])
    b = x_train.T.dot(y_train)
    Wrr = np.linalg.inv(a).dot(b)
    
    return Wrr


def predict(X_test, Wrr):
    """ Predict using Ridge Regression and predicts the respective Y_test values

    This function  takes 2 inputs. `x_test`, `Wrr`. 

    Parameters
    ----------
    x_test : numpy-array
        a dataset represented as a numpy-array.
    Wrr : numpy-array
        a dataset represented as a numpy-array.

    Returns
    -------
    numpy-array
    Y_predict: predicted values for the test data test.
    """

    Y_predict = X_test.dot(Wrr)
    
    return Y_predict


def RMSE(Y_predict, Y_test, Lamda):
    rmse = []
    
    for i in Lamda:
        a = math.sqrt(np.sum((Y_predict[i] - Y_test)**2) / 42)
        rmse.append(a)
        
    return  rmse


#load train data as numpy array
X_train = np.genfromtxt('/Users/Amiros/Dropbox/University/4- Spring 2017/ML/homework/hw1-data/X_train.csv',delimiter=',')
Y_train = np.genfromtxt('/Users/Amiros/Dropbox/University/4- Spring 2017/ML/homework/hw1-data/Y_train.csv',delimiter=',')
# X_train = np.delete(X_train,-1,1)

#load test data as numpy array
X_test = np.genfromtxt('/Users/Amiros/Dropbox/University/4- Spring 2017/ML/homework/hw1-data/X_test.csv',delimiter=',')
Y_test = np.genfromtxt('/Users/Amiros/Dropbox/University/4- Spring 2017/ML/homework/hw1-data/Y_test.csv',delimiter=',')
# X_test = np.delete(X_test,-1,1)


## Part A

# 1-A
lmda_1A = np.arange(5001)
w_1A = []

#fitting RR on X_train
for i in lmda_1A:
    w_1A.append(fit(X_train, Y_train, i))
w_1A = np.array(w_1A)

#SVD of X_train
U, s, V = np.linalg.svd(X_train, full_matrices= 0)

#Calculating df(lmda)
df_1A = []
for i in lmda_1A:
    a = sum((s**2/(i + s**2)))
    df_1A.append(a)
#     df.reverse
df_1A = np.array(df_1A)


plt.figure(figsize=(12,8))

color = ['#1b85b8', '#5a5255', '#559e83', '#ae5a41', '#c3cb71', '#BCBA44', 'k']
feature = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7']
for i in range(w_1A.shape[1]):
    plt.scatter(df_1A, w_1A[:,i],
            marker='o',
            color=color[i],
            alpha=0.3,
            s = 10,
            label=feature[i])
    
# Chart title
plt.title('Ridge Regression Coefficeints vs f(lambda)')

# y label
plt.ylabel('Coefficients')

# x label
plt.xlabel('df(λ)')

# and a legend
plt.legend(loc='lower left')

# ## 1-c

lmda_1C = np.arange(51)
Y_predict = []

#predict Y
for i in lmda_1C:
    a = fit(X_train, Y_train, i)
    Y_predict.append(predict(X_test,a))
Y_predict = np.array(Y_predict)


#plotting root mean squared error (RMSE)
rmse = RMSE(Y_predict, Y_test, lmda_1C)
    
plt.figure(figsize=(12,8))

plt.scatter(lmda_1C, rmse,
            marker='o',
            color='#89B8FF',
            alpha=0.8,
            s = 15)
   
# Chart title
plt.title('RMSE plot')

# y label
plt.ylabel('RMSE value')

# x label
plt.xlabel('Lambda')


## Part 2

# Create 2nd and 3rd degree matrix
X_train_1st = np.delete(X_train,-1,1)
X_train_2nd = np.concatenate((X_train, np.square(X_train_1st)), axis=1)
X_train_3rd = np.concatenate((X_train_2nd, np.power(X_train_1st, 3)), axis=1)

X_test_1st = np.delete(X_test,-1,1)
X_test_2nd = np.concatenate((X_test, np.square(X_test_1st)), axis=1)
X_test_3rd = np.concatenate((X_test_2nd, np.power(X_test_1st, 3)), axis=1)

lmda_2 = np.arange(501)

Y_predict_P1 = []
Y_predict_P2 = []
Y_predict_P3 = []


#predicting Y for 1st, 2nd, and 3rd order polynomials
for i in lmda_2:    
    # 1st order polynomial
    fit_p1 = fit(X_train, Y_train, i)
    Y_predict_P1.append(predict(X_test,fit_p1))
    
    # 2nd order polynomial
    fit_p2 = fit(X_train_2nd, Y_train, i)
    Y_predict_P2.append(predict(X_test_2nd,fit_p2))
    
    
    # 3rd order polynomial
    fit_p3 = fit(X_train_3rd, Y_train, i)
    Y_predict_P3.append(predict(X_test_3rd,fit_p3))
       
    
Y_predict_P1 = np.array(Y_predict_P1)
Y_predict_P2 = np.array(Y_predict_P2)
Y_predict_P3 = np.array(Y_predict_P3)

# Calculating root mean squared error (RMSE) for all 3 degrees
rmse_P1 = RMSE(Y_predict_P1, Y_test, lmda_2)
rmse_P2 = RMSE(Y_predict_P2, Y_test, lmda_2)
rmse_P3 = RMSE(Y_predict_P3, Y_test, lmda_2)


# Plotting RMSE for all 3 degrees
    
plt.figure(figsize=(12,8))

plt.scatter(lmda_2, rmse_P1,
            marker='o',
            color='#e1f7d5',
            alpha=0.8,
            s = 5,
            label="1st degree polynomial")

plt.scatter(lmda_2, rmse_P2,
            marker='o',
            color='#ffbdbd',
            alpha=0.8,
            s = 5,
            label="2nd degree polynomial")

plt.scatter(lmda_2, rmse_P3,
            marker='o',
            color='#c9c9ff',
            alpha=0.8,
            s = 5,
            label="3rd degree polynomial")



plt.ylim([0, 3.5])
plt.xlim([0, 501])

plt.axvline(x=np.argmin(rmse_P2))
plt.xticks([np.argmin(rmse_P2)])
    
# Chart title
plt.title('RMSE plot')

# y label
plt.ylabel('RMSE value')

# x label
plt.xlabel('Lambda')

# legend
plt.legend(loc='lower right')

