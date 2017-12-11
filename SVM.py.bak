# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:40:17 2017

@author: dell
"""

import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat
    
def showDataSet(dataMat, labelMat):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], c = 'red')
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], c = 'blue') #负样本散点图
    plt.show()
    
if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    showDataSet(dataMat, labelMat)