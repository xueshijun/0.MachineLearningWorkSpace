# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on 2016年1月17日
#Version: 1.0
#-------------------------------------------------------------------------------

'''
使用k-近邻算法的手写识别系统
该数据集合修 改 自 “手写数字数据集的光学识别”一文中的数据集合，该文登载于2010年10月3日的UCI机器学习
资料库中http://archive.ics.uci.edu/ml。
(1)收集数据：提供文本文件。
(2)准备数据：编写函数classify0(),将图像格式转换为分类器使用的制格式。
(3)分析数据：在?5^0^命令提示符中检查数据，确保它符合要求。
(4)训练算法：此步驟不适用于各近邻算法。
(5)测试算法：编写函数使用提供的部分数据集作为测试样本，测试样本与非测试样本的区别在于测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。
(6)使用算法：本例没有完成此步驟，若你感兴趣可以构建完整的应用程序，从图像中提取数字，并完成数字识别，美国的邮件分拣系统就是一个实际运行的类似系统。
'''

#准备数据：将图像转换为测试向量
#目录trainingDigits中包含了大约2000个例子，每个数字大约有200个样本；
#目录testDigits中包含了大约900个测试数据。
#将图像格式化处理为一个向量。我们将把一个32 x 32的二进制图像矩阵转换为1 x 1024的向量，
#这样前两节使用的分类器就可以处理数字图像信息了。

from numpy import *
from os import listdir

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

def img2vector(filename):
    returnVector=zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline();
        for j in range(32):
            returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector

            
            
testVector=img2vector('G:/0.MachineLearning/0.WorkSpace/MachingLearningInAction/com/kNN/digits/testDigits/0_13.txt')
print testVector[0,0:31]


def handWritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat=zeros((m,1024));
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector(('digits/trainingDigits/%s')%(fileNameStr))

    testFileList = listdir('digits/testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest= img2vector(('digits/testDigits/%s')%(fileNameStr))
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print (('[classifier:%10s][real:%10s]')%(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errorCount+=1.0
    print(('error number %5s')%(errorCount))
    print(('error number %5s')%(errorCount/float(mTest)))


            
            
            
handWritingClassTest();
            



