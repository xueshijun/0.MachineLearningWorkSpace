# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Sun Feb 21 09:17:23 2016
#Version: 1.0
#-------------------------------------------------------------------------------
'''
K-均值聚类
优 点 ：容易实现。
缺 点 ：可能收敛到局部最小值，在大规模数据集上收敛较慢。
适用数据类型：数值型数据。

K-均值聚类的一般流程
(1)收集数据：使用任意方法。
(2)准备数据：需要数值型数据来计算距离，也可以将标称型数据映射为二值型数据再用于距离计算。
(3)分析数据：使用任意方法。
(4)训练算法：不适用于无监督学习，即无监督学习没有训练过程。
(5)测试算法：应用聚类算法、观察结果。可以使用量化的误差指标如误差平方和(后面会介绍)来评价算法的结果。
(6)使用算法：可以用于所希望的任何应用。通常情况下，簇质心可以代表整个簇的数据来做出决策。

伪代码：
创建女个点作为起始质心（ 经常是随机选择）
当任意一个点的簇分配结果发生改变时
    对数据集中的每个数据点
        对每个质心
            计算质心与数据点之间的距离
        将数据点分配到距其最近的簇
    对每一个簇，计算簇中所有点的均值并将均值作为质心
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat
#计算欧氏距离
def distEclud(vecFrom, vecTo):
    return sqrt(sum(power(vecFrom - vecTo, 2))) #la.norm(vecA-vecB)
#为给定数据集构建一个包含K个随机质心的集合。
#随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一组的最小和最大值来完成。
def randCentroids(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]);  maxJ=max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids
    
dataMat = mat(loadDataSet('testSet.txt'))
min(dataMat[:,0]);min(dataMat[:,1])
max(dataMat[:,0]);max(dataMat[:,1])
#生成min和max之间的值：
print randCentroids(dataMat,2) 
#距离计算
print distEclud(dataMat[0],dataMat[1])
print '---------------'
'''
该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
这个过程重复数次，直到数据点的簇分配结果不再改变为止。
'''
def kMeans(dataSet, k, distMeas=distEclud, createCentroids=randCentroids):
    m = shape(dataSet)[0]
    #簇分配结果矩阵:一列记录簇索引值，第二列存储误差
    clusterAssment = mat(zeros((m,2)))
    centroids = createCentroids(dataSet, k)
    clusterChanged = True
    while clusterChanged:#该值为true，则继续迭代。
        clusterChanged = False
        #遍历所有数据找到距离每个点最近的质心
        for i in range(m): 
            minDist = inf; minIndex = -1
            #对每个点遍历所有质心并计算点到每个质心的距离
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print centroids
        for cent in range(k):#遍历所有质心并更新它们的取值
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment#返回所有的类质心与点分配结果
    
 
myCentroids,clusterAssment=kMeans(dataMat,4)
print '---------------'
print myCentroids
        
'''
聚类的目标是在保持族数目不变的情况下提高簇的质量
一种方法是将具有最大SSE值的簇划分成两个簇。
即将最大簇包含的点过滤出来并在这些点上运行K-均值,为了保持簇总数不变，可以将某两个簇进行合并。

有两种可以量化的办法：
    1)合并最近的质心:计算所有质心之间的距离，然后合并距离最近的两个点来实现
    2)合并两个使得SSE增幅最小的质心:合并两个簇然后计算总SSE值.
        必须在所有可能的两个簇上重复上述处理过程,直到找到合并最佳的两个簇为止。
'''


'''
二分K-均值算法的伪代码形式:
将所有点看成一个簇
当簇数目小于k时
    对于每一个簇
        计算总误差
        在给定的簇上面进行K-均值聚类(k=2)
        计算将该簇一分为二之后的总误差
    选择使得误差最小的那个簇进行划分操作

注:另一种做法是选择88£最大的簇进行划分，直到簇数目达到用户指定的数目为止。
'''
def binKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0] ;clusterAssment = mat(zeros((m,2)))
    #创建一个初始簇
    centroid0 = mean(dataSet, axis=0).tolist() #axis = 0表示沿矩阵的列方向进行均值计算
    centList =[centroid0] #簇列表
    for j in range(m):#calc initial Error  
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:]) 
 
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):#遍历所有的簇来决定最佳的簇进行划分
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment
 
dataMat = loadDataSet('testSet2.txt')
#centList, clusterAssment =
centList , clusterAssmentbinKmeans(mat(dataMat),3)
'''

'''
