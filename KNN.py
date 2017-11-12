# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 08:41:47 2017

@author: dell
"""

###################KNN
import numpy as np

def createDataSet():
    group = np.array([[1,10],[5,89],[108,5],[115,8]])
    labels = ['爱情片','爱情片','动作片','动作片']
    return group,labels

if __name__ == '__main__':
    group,labels = createDataSet()
    print(group)
    print(labels)

#########################
import numpy as np
import operator

"""
function:KNN alogrithm classifier

parameters:
    inX - testing dataset
    dataSet - training dataset
    labels - classifier labels
    k - select min K 

returns:
    classifier result
"""

def classify0(inX, dataSet, labels, k):
    #row num
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    #创建数据集
    group, labels = createDataSet()
    #测试集
    test = [101,20]
    #kNN分类
    test_class = classify0(test, group, labels, 3)
    #打印分类结果
    print(test_class)