# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on 2015年8月30日
#Version: 1.0
#-------------------------------------------------------------------------------
'''
Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

k-近邻算法是基于实例的学习，使用算法时我们必须有接近实际数据的训练样本数据。
k-近邻算法必须保存全部数据集，如果训练数据集的很大，必须使用大量的存储空间。
此外,由于必须对数据集中的每个数据计算距离值，实际使用时可能非常耗时。
k-近邻算法的另一个缺陷是它无法给出任何数据的基础结构信息，因此我们也无法知晓平均实例样本和典型实例样本具有什么特征。

k决策树就是k-近邻算法的优化版，可以节省大量的存储空间和计算时间。
'''


'''kNN算法的一般流程
(1)收集数据：可以使用任何方法。
(2)准备数据：距离计算所需要的数值，最好是结构化的数据格式。
(3)分析数据：可以使用任何方法。
(4)训练算法：此步驟不适用于1 近邻算法。
(5)测试算法：计算错误率。
(6)使用算法：首先需要输入样本数据和结构化的输出结果，然后运行女-近邻算法判定输
入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理。
'''
from numpy import *
import operator



#准备数据
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

group,labels=createDataSet()

'''
(1)计算已知类别数据集中的点与当前点之间的距离；
(2)按照距离递增次序排序；sqrt((x1-x2)^2+(y1-y2)^2)
(3)选取与当前点距离最小的k个点；
(4)确定前k个点所在类别的出现频率；
(5)返回前k个点出现频率最高的类别作为当前点的预测分类。
'''
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


print classify0([0,0], group, labels, 3)






