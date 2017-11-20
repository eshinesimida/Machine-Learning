# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 20:45:43 2017

@author: dell
"""

def Gradient_Ascent_test():
    def f_prime(x_old):
        return -2 * x_old + 4
    x_old = -1
    x_new = 0 
    alpha = 0.01
    presision = 0.00000001
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    print(x_new)
    
if __name__ == '__main__':
    Gradient_Ascent_test()
    
#############################
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat

def plotDataSet():
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    #fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's', alpha=.5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('dataset', dpi = 600)
    plt.show()
    

if __name__ == '__main__':
    plotDataSet()
    
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
    
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    return weights.getA(), weights_array
    
if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    print(gradAscent(dataMat, labelMat))
    
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    #fig = plt.figure()
    ax = plt.subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig('bestfit', dpi = 600)
    plt.show()
    
if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()           
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights)

############################
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    weights_array = np.array([])
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha *error*dataMatrix[randIndex]
            weights_array = np.append(weights_array, weights, axis = 0)
            del(dataIndex[randIndex])
    weights_array = weights_array.reshape(numIter*m, n)
    return weights, weights_array

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = stocGradAscent1(np.array(dataMat), labelMat)
    plotBestFit(weights)
            
def plotWeights(weights_array1,weights_array2):
    font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size = 14)
    fig, axs = plt.subplots(nrows = 3, ncols = 2, sharex = False, sharey = False, figsize = (20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    axs[0][0].plot(x1, weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系', FontProperties = font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', FontProperties = font)
    plt.setp(axs0_title_text, size = 20, weight = 'bold', color = 'black')
    plt.setp(axs0_ylabel_text, size = 20, weight = 'bold', color = 'black')
     #绘制w1与迭代次数的关系
    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
     #绘制w2与迭代次数的关系
    axs[2][0].plot(x1,weights_array1[:,2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    x2 = np.arange(0, len(weights_array2), 1)
    #绘制w0与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:,0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    
    plt.savefig('logical', dpi= 600)

    plt.show()       
    
if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()           
    weights1,weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)

    weights2,weights_array2 = gradAscent(dataMat, labelMat)
    plotWeights(weights_array1, weights_array2)
    