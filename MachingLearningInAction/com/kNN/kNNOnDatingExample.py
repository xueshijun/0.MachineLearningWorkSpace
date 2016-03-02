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
(1)收集数据：提供文本文件:datingTestSet.txt。
    海伦的样本主要包含以下3种特征：
    □ 每年获得的飞行常客里程数
    □ 玩视频游戏所耗时间百分比
    □ 每周消费的冰淇淋公升数
(2)准备数据：使用Python解析文本文件。
(3)分析数据：使用matplotlib画二维扩散图。
(4)训练算法：此步驟不适用于k近邻算法。
(5)测试算法：使用海伦提供的部分数据作为测试样本。
测试样本和非测试样本的区别在于：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。
(6)使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。
'''
from numpy import *
import operator

#准备数据：从文本文件中解析数据
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         # 1）打开文件，得到文件的行数
    returnMat = zeros((numberOfLines,3))        # 2）创建以零填充的矩阵
    classLabelVector = []                       # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():                 # 3）循环处理文件中的每行数据
        line = line.strip()                     # 截取所有回撤字符，并用tabk字符\t将上一步得到的整行数据
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat,classLabelVector


#准备数据：归一化数值:可以将任意取值范围的特征值转化为0到 1区间内的值：
'''newValue = (oldValue - min )/(max - min)
'''
def autoNorm(dataSet):
    minValues=dataSet.min(0)
    maxValues=dataSet.max(0)
    ranges=maxValues-minValues
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minValues,(m,1))
    #tile()函数将变量内容复制成输人矩阵同样大小的矩阵，注意这是具体特征值相除
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minValues



 
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



 
#normMat,ranges,minValue = autoNorm(datingDataMat)

def datingClassTest():
    hoRatio =0.10
    datingDataMat,datingLabels=file2matrix('G:/0.MachineLearning/0.WorkSpace/MachingLearningInAction/com/kNN/datingTestSet.txt')
    normMat,ranges,minValues=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print(('[the classifier:%10s][the real answer:%10s]')%(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print (("the total error rate is:%f")%(errorCount/float(numTestVecs)))
    
datingClassTest();


'''绘图'''
datingDataMat,datingLabels=file2matrix('G:/0.MachineLearning/0.WorkSpace/MachingLearningInAction/com/kNN/datingTestSet.txt')

import matplotlib
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
#15.0*array(datingLabels),15.0*array(datingLabels)
plt.show()

def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    '''
    percentTats=float(raw_input("percentage of time spent playing video games？"))
    ffMiles=float(raw_input("frequent flier miles earned per year？"))    
    iceCream=float(raw_input("liters of ice cream comsumed per year？"))
    '''
    percentTats=40910
    ffMiles=8.326976
    iceCream=0.953952
    
    datingDataMat,datingLabels=file2matrix('G:/0.MachineLearning/0.WorkSpace/MachingLearningInAction/com/kNN/datingTestSet2.txt')
    normMat,ranges,minValues=autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minValues)/ranges,normMat,datingLabels,3)
    #print classifierResult
    print (('you will probable like this person: %s ') % ( resultList[int(classifierResult)-1]))
 
    
classifyPerson()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    