# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 20:10:32 2018

@author: Administrator
"""
import numpy as np
from numpy import *
import time

import matplotlib.pyplot as plt
fileName = 'DL/testSet.txt'
def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=[float(s) for s in curLine]
        dataMat.append(fltLine)
    return dataMat
 
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))
 
def randCent(dataSet,k):
    n=shape(dataSet)[1]
    centroids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j])-minJ)
        centroids[:,j]=mat(minJ+rangeJ*random.rand(k,1))  #k个随机数
    return centroids


datMat = mat(loadDataSet('DL/testSet.txt')) 

randCent(datMat, 2)
dataSet = datMat
distEclud(datMat[0], datMat[1])

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=inf
            minIndex=-1
            for j in range(k):
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                if distJI<minDist:
                    minDist=distJI
                    minIndex=j 
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:]=mean(ptsInClust,axis=0)
    return centroids,clusterAssment


myCen, clustAssing = KMeans(datMat, 4)

import matplotlib.gridspec as gridspec
def showCluster(dataSet, k, centroids, clusterAssment):
	numSamples, dim = dataSet.shape
	if dim != 2:
		print ("Sorry! I can not draw because the dimension of your data is not 2!")
		return 1
 
	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
	if k > len(mark):
		print ("Sorry! Your k is too large! please contact Zouxy")
		return 1
 
	# draw all samples

	for i in range(numSamples):
		markIndex = int(clusterAssment[i, 0])
		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
 
	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
	# draw the centroids
	for i in range(k):
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)
  
	plt.show()
 

  

showCluster(dataSet, k, myCen, clustAssing)            
