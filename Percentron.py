# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 17:03:33 2018

@author: admin
"""

import numpy as np
class Percentron(object):
    def __init__(self, eta = 0.1, n_iter  = 10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
print(df.tail())

import matplotlib.pyplot as plt
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values
plt.scatter(X[:50,0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100,0], X[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('sepel length')
plt.show()

ppn = Percentron(eta = 0.1, n_iter = 10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassfications')
plt.show()


from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx= None, resolution=0.02):
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^','v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap= ListedColormap(colors[:len(np.unique(y))])
    
    #plot the decison surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1 , X[:, 1].max() + 1
    
    xx1,xx2 = np.meshgrid(np.arange(x1_min, x1_max,resolution), np.arange(x2_min, x2_max,resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x= X[y==cl, 0], y = X[y==cl, 1],
            alpha=0.8, c=cmap(idx), marker= markers[idx],label = cl)
######################################################################################################
plot_decision_regions(X, y, classifier = ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()


from sklearn import datasets
class AdalineGD(object):
    def __init__(self, eta = 0.01, n_inter = 50):
        self.eta = eta
        self.n_inter = n_inter
        
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_inter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] = self.eta*X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/ 2.0
            self.cost_.append(cost)
            
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return self.net_input(X)
    
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
ada1 = AdalineGD(eta=0.01,n_inter=10).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - learning rate 0.01')        
ada2 = AdalineGD(eta=0.0001,n_inter=10).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline - learning rate 0.0001') 

X_std = np.copy(X)
X_std[:,0] = (X[:, 0] - X[:,0].mean()) / X[:, 0].std()
X_std[:,1] = (X[:, 1] - X[:,1].mean()) / X[:, 1].std()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
ada3 = AdalineGD(eta=0.01,n_inter=10).fit(X_std, y)
ax[0].plot(range(1, len(ada3.cost_) + 1), np.log10(ada3.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - learning rate 0.01 after standerdized') 
ada4 = AdalineGD(eta=0.0001,n_inter=10).fit(X_std, y)
ax[1].plot(range(1, len(ada4.cost_) + 1), np.log10(ada4.cost_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline - learning rate 0.0001 after standerdized') 

from numpy.random import seed

class AdalineSGD(object):
    def __init__(self, eta=0.01, n_inter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_inter = n_inter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
            
        
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_inter):
            if self.shuffle:
                X, y = self._shuffle(X,y)
                
            cost = []
            #kk=0
            #print("*****************************")
            #print(len(y))
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi, target))
                #kk += 1
                #print(kk)
            
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
             
        return self
    
    '''
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
            
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
            
        return self
    '''
    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]
    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        #print(error)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
       
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return self.net_input(X)
    
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
# 特征缩放，能有助于收敛；但不是说之前不收敛的，特征缩放后一定能收敛；
X_std = np.copy(X)
X_std[:,0] = (X[:, 0] - X[:,0].mean()) / X[:, 0].std()
X_std[:,1] = (X[:, 1] - X[:,1].mean()) / X[:, 1].std()
ada = AdalineSGD(n_inter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Averrage Cost')
plt.show()


##############################################################################
def plot_decision_regions(X, y, classifier, test_idx= None, resolution=0.02):
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^','v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap= ListedColormap(colors[:len(np.unique(y))])
    
    #plot the decison surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1 , X[:, 1].max() + 1
    
    xx1,xx2 = np.meshgrid(np.arange(x1_min, x1_max,resolution), np.arange(x2_min, x2_max,resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x= X[y==cl, 0], y = X[y==cl, 1],
            alpha=0.8, c=cmap(idx), marker= markers[idx],label = cl)
    
    #highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:,1], c='',alpha = 1.0, linewidth = 1, marker = '0', s = 55, label = 'test set')
##############################################################################
plot_decision_regions(X_std, y, classifier = ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standerdized]')
plt.ylabel('petal length [standerdized]')