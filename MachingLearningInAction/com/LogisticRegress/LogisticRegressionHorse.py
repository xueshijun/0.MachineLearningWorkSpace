# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Sun Jan 24 23:11:01 2016
#Version: 1.0
#-------------------------------------------------------------------------------
from numpy import *
'''
示例:使用Logistic回归估计马疝病的死亡率
(1)收集数据：给定数据文件。
        http://archive.ics.uci.edu/ml/datasets/Horse+Colic
(2)准备数据：用Python解析文本文件并填充缺失值。
    若机器上的某个传感器损坏导致一个特征无效时,不需要扔掉整个数据,
    因为有时候数据相当昂贵， 扔掉和重新获取都是不可取的,采用一些方法：
        □使用可用特征的均值来填补缺失值；
        □使用特殊值来填补缺失值，如-1;
        □忽略有缺失值的样本；
        □使用相似样本的均值添补缺失值；
        □使用另外的机器学习算法预测缺失值。


    在预处理阶段需要做两件事： 
        第一，所有的缺失值必须用一个实数值来替换，因为我们使用的NumPy数据类型不允许包含缺失值。
            这里选择实数0来替换所有缺失值，恰好能适用于Logistic回归。
            这样做的直觉在于 ，我们需要的是一个在更新时不会影响系数的值。
            回归系数的更新公式如下
                weights = weights + alpha * error * dataMatrix[randIndex]
            如果dataMatrix的某特征对应值为0，那么该特征的系数将不做更新:
                weights = weights
            由于sigmoid(0)=0.5，即它对结果的预测不具有任何倾向性，因此上述做法也不会对误差项造成任何影响。
            基于上述原因，将缺失值用0代替既可以保留现有数据，也不需要对优化算法进行修改。
            此外，该数据集中的特征取值一般不为0 , 因此在某种意义上说它也满足“特殊值”这个要求。
        第二件事是，如果在测试数据集中发现了一条数据的类别标签已经缺失，
            那么我们的简单做法是将该条数据丢弃。这是因为类别标签与特征不同，^艮难确定采用某个合适的值来替换。
            采用Logistic回归进行分类时这种做法是合理的，而如果采用类似kNN的方法就可能不太可行。
    
        horseColicTest.txt和horseColicTraining.txt
(3)分析数据：可视化并观察数据。
(4)训练算法：使用优化算法，找到最佳的系数。
(5)测试算法：为了量化回归的敢果，需要观察错误率。
    根据错误率决定是否回退到训练阶段，通过改变迭代的次数和步长等参数来得到更好的回归系数。
(6)使用算法：实现一个简单的命令行程序来收集马的症状并输出预测结果并非难事，这可以做为留给读者的一道习题。
除了部分指标主观和难以测量外，该数据还存在一个问题，数据集中有30%的值是缺失的。
下面将首先介绍如何处理数据集中的数据缺失问题，然 后 再 利 用Logistic回归和随机梯度上升算法来预测病马的生死。
'''
import LogisticRegression
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    '''计算回归系数向量这里可以自由设定迭代的次数，例如在训练集
上使用500次迭代，实验结果表明这比默认迭代150次的效果更好。'''
    trainWeights = LogisticRegression.stocGranAscentImprove(array(trainingSet), trainingLabels, 1000)
    '''在系数计算完成之后，导人测试集并计算分类错误率'''
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate
colicTest()
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))

multiTest()