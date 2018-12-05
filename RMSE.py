# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:54:23 2018

@author: Administrator
"""

import numpy as np
import pylab

x= np.linspace(-1, 1, 100)
signal = 2 + x + 2*x*x
noise = np.random.normal(0, 0.1, 100)
y = signal + noise
pylab.plot(signal, 'b')
pylab.plot(y, 'g')
pylab.plot(noise, 'r')
pylab.xlabel('x')
pylab.ylabel('y')
pylab.legend(["without noise","with noise",'noise'], loc = 2)

x_train = x[0:80]
y_train = y[0:80]

#model with degree 1

pylab.figure()
degree = 2
X_train = np.column_stack([np.power(x_train,i) for i in range(0, degree)])
model = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(), X_train)), X_train.transpose()),
               y_train)
#model = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(),X_train)),X_train.transpose()),
          #        y_train)
pylab.plot(x, y, 'g')
pylab.xlabel('x')
pylab.ylabel('y')
predicted = np.dot(model, [np.power(x,i) for i in range(0, degree)])
pylab.plot(x, predicted, 'r')
pylab.legend(["Actual",'predicted'], loc = 2)
train_rmse1 = np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80] )))
test_rmse1 = np.sqrt(np.sum(np.dot(y[80:] - predicted[80:], y[80:] - predicted[80:] )))
print('Train RMSE (degree = 1)', train_rmse1)
print('Test RMSE (degree = 1)', test_rmse1)

##degree =2
pylab.figure()
degree = 3
X_train = np.column_stack([np.power(x_train,i) for i in range(0,degree)])
model = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(),X_train)),
X_train.transpose()),y_train)
pylab.plot(x,y,'g')
pylab.xlabel("x")
pylab.ylabel("y")
predicted = np.dot(model, [np.power(x,i) for i in range(0,degree)])
pylab.plot(x, predicted,'r')
pylab.legend(["Actual", "Predicted"], loc = 2)
train_rmse1 = np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80],
y_train - predicted[0:80])))
test_rmse1 = np.sqrt(np.sum(np.dot(y[80:] - predicted[80:],
y[80:] - predicted[80:])))
print("Train RMSE (Degree = 2)", train_rmse1)
print("Test RMSE (Degree = 2)", test_rmse1)

##degree =8
pylab.figure()
degree = 9
X_train = np.column_stack([np.power(x_train,i) for i in range(0,degree)])
model = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(),X_train)),
X_train.transpose()),y_train)
pylab.plot(x,y,'g')
pylab.xlabel("x")
pylab.ylabel("y")
predicted = np.dot(model, [np.power(x,i) for i in range(0,degree)])
pylab.plot(x, predicted,'r')
pylab.legend(["Actual", "Predicted"], loc = 3)
train_rmse1 = np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80],
y_train - predicted[0:80])))
test_rmse1 = np.sqrt(np.sum(np.dot(y[80:] - predicted[80:],
y[80:] - predicted[80:])))
print("Train RMSE (Degree = 8)", train_rmse1)
print("Test RMSE (Degree = 8)", test_rmse1)