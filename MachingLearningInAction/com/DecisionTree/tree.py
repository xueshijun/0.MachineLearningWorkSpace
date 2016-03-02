#coding=utf-8
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on 2015年8月30日
#Version: 1.0
#-------------------------------------------------------------------------------
from math import log
import operator
'''
决策树是一种贪心算法，它要在给定时间内做出最佳选择，但并不关心能否达到全局最优。
决策树的一般流程
(1)收集数据：可以使用任何方法。
(2)准备数据：树构造算法只适用于标称型数据，因此数值型数据必须离散化。
(3)分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期。
(4)训练算法：构造树的数据结构。
(5)测试算法：使用经验树计算错误率。
(6)使用算法：此步骤可以适用于任何监督学习算法，而使用决策树可以更好地理解数据的内在含义。
'''

'''
创建分支的伪代码函数createBranch如下所示：
检测数据集中的每个子项是否属于同一分类：
    If so
        return 类标签；
    Else
        寻找划分数据集的最好特征
        划分数据集
        创建分支节点
            for 每个划分的子集
                调用函数createBranch并增加返回结果到分支节点中
        return 分支节点
'''

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers','fish?']
    return dataSet, labels

myData,labels=createDataSet()
print labels            #['flippers', 'fish?']

#熵计算
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts={}
    for featVec in dataSet:#the the number of unique elements and their occurance
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob = float( labelCounts[key] ) / numEntries 
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#print calcShannonEnt(myData)

#待划分的数据集，划分数据集的特征，特征的返回值
'''
a=[1,2,3]
b=[3,4,5]
a.append(b) #[1,2,3,[4,5,6]]

a.extend(b) #[1,2,3,4,5,6]
'''
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature= -1;
    for i in range(numFeatures):
        featList =[example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            #计算每种划分的信息熵
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy +=prob * calcShannonEnt(subDataSet)
        infoGain =baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
            
    

print "bestFeature ",chooseBestFeatureToSplit(myData)

#返回出现次数最多的分类名
'''如果数据集已经处理了所有属性,但是类标签依然不是唯一的,
我们通常会采用多数表决的方法决定该叶子节点的分类。'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reversed=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    featValues = [example[bestFeature] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels)
    return myTree

tree = createTree(myData,labels)    
print tree                  #{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
print tree.keys()   #['no surfacing']
print tree [tree.keys()[0]]                  #{0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}

'''
ID3是一个好的算法但并不完美。ID3算法无法直接处理数值型数据，
尽管我们可以通过量化的方法将数值型数据转化为标称型数值，但是如果存在太多的特征划分，ID3算法仍然会面临其他问题。
'''
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) #用index方法查找当前列表中第一个匹配firstStr变量的元素
    for key in secondDict.keys():
        if(testVec[featIndex]==key):
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel =secondDict[key]
    return classLabel

print classify(tree,labels,[0,1])
print classify(tree,labels,[1,0])
print classify(tree,labels,[1,1])


''''决策树的存储，
#构造决策树是很耗时的任务，即使处理很小的数据集
使用Python模块pickle序列化对象
序列化对象可以在磁
盘上保存对象，并在需要的时候读取出来。任何对象都可以执行序列化操作， 字典对象也不例外。
'''
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

storeTree(tree,'classifierStorage.txt')
print grabTree('classifierStorage.txt')

