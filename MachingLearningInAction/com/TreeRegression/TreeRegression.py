# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Fri Feb 19 21:25:45 2016
#Version: 1.0
#-------------------------------------------------------------------------------
'''
若叶节点使用的模型是分段常数则称为回归树，
若叶节点使用的模型是线性回归方程则称为模型树。

CART算法可以用于构建二元树并处理离散型或連续型数据的切分。
若使用不同的误差准则，就可以通过CART算法构建模型树和回归树。

两种剪枝方法分别:
    预剪枝(在树的构建过程中就进行剪枝)和后剪枝(当树构建完毕再进行剪枝)，预剪枝更有效但需要用户定义一些参数。
    
    
树回归
优点：可以对复杂和非线性的数据建模。
缺点：结果不易理解。
适用数据类型：数值型和标称型数据

树回归的一般方法
(1) 收集数据：采用任意方法收集数据。
(2) 准备数据：需要数值型的数据，标称型数据应该映射成二值型数据。
(3) 分析数据：绘出数据的二维可视化显示结果，以字典方式生成树。
(4) 训练算法：大部分时间都花费在叶节点树模型的构建上。
(5) 测试算法：使用测试数据上的R2值来分析模型的效果。
(6) 使用算法：使用训练出的树做预测，预测結果还可以用来做很多事情
'''

from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t') #读取一个以tab为分隔符的文件
        fltLine = map(float,curLine)       #每行的内容保存成一组浮点数
        dataMat.append(fltLine)
    return dataMat
'''
#通过数组过滤方式将上述数据集合切分得到两个子集并返回
dataSet：数据集
feature：待切分的特征
value ：根据该特征进行切分的阈值
'''
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0,mat1

testMat = mat(eye(4))
mat0,mat1=binSplitDataSet(testMat,2,0)
print mat0,mat1


''' 
伪代码：
对每个特征：
    对每个特征值：
        将数据集切分成两份.
        计算切分的误差
        如果当前误差小于当前最小误差， 那么将当前切分设定为最佳切分并更新最小误差
    返回最佳切分的特征和阈值
'''
#负责生成叶节点
def regLeaf(dataSet):
    return mean(dataSet[:,-1])
#误差估计函数
#总方差=用均方差乘以数据集中样本的个数
def regErr(dataSet):#直接调用均方差函数var()
    return var(dataSet[:,-1]) * shape(dataSet)[0]
#找到数据的最佳二元切分方式 
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0];#容许的误差下降值
    tolN = ops[1] #切分的最少样本数
    
    #统计不同剩余特征值的数目
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #如果误差减少不大则退出
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    #如果切分出的数据集很小则退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split



#基于CART算法构建回归树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat#特征
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree   
myMat=mat(loadDataSet('ex00.txt'))
print createTree(myMat)
myMat=mat(loadDataSet('ex0.txt'))
print createTree(myMat)

'''
树剪枝
通过降低决策树的复杂度来避免过拟合的过程称为剪枝(pruning)。
预剪枝:
在函数chooseBestSplit()的提前终止条件，实际上是在进行一种所谓的预剪枝(prepruning)操作。
'''

#myMat=mat(loadDataSet('ex2.txt'))
#print createTree(myMat,ops=(0,1))#构建的树过于臃肿，它甚至为数据集中每个样本都分配了一个叶节点。
#print createTree(myMat,ops=(10000,4))#仅有两个叶节点组成的树
'''
后剪枝(postpruning)：
需要使用测试集和训练集,首先指定参数,使得构建出的树足够大、足够复杂，便于剪枝。
接下来从上而下找到叶节点,用测试集来判断将这些叶节点合并是否能降低测试误差。如果是的话就合并。

伪代码:
基于已有的树切分测试数据：
    如果存在任一子集是一棵树，则在该子集递归剪枝过程
    计算将当前两个叶节点合并后的误差
    计算不合并的误差
    如果合并会降低误差的话，就将叶节点合并
'''
#于判断当前处理的节点是否是叶节点。
def isTree(obj):
    return (type(obj).__name__=='dict')
#它从上往下遍历树直到叶节点为止
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    #如果找到两个叶节点则计算它们的平均值。
    #对树进行塌陷处理（ 即返回树平均值)，
    return (tree['left']+tree['right'])/2.0 
   
    
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print "merging"; return treeMean
        else: return tree
    else: return tree
    



#创建所有可能中最大的树
myMat=mat(loadDataSet('ex2.txt'))
myTree = createTree(myMat,ops=(0,1))#构建的树过于臃肿，它甚至为数据集中每个样本都分配了一个叶节点。
#导人测试数据
myDataTest=loadDataSet('ex2test.txt')
print prune(myTree,mat(myDataTest))

'''
模型树
把叶节点设定为分段线性函数，这里所谓的分段线性（ 扭& — 北丨丨目1 ) 是指模型由多个线性片段组成。
特点:
    可解释性是它优于回归树;
    模型树也具有更髙的预测准确度。
'''
print '==============================='

#主要功能是将数据集格式化成目标变量Y和自变量X
def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    #将X与Y中的数据格式化
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    
    xTx = X.T*X
    #如果矩阵的逆不存在也会造成程序异常。
    if linalg.det(xTx) == 0.0:raise NameError('matrix is singular,cannot do inverse,try increasing the second value of ops')
    ws = xTx.I * (X.T * Y) 
    return ws,X,Y
#
def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws
#在给定的数据集上计算yHat和Y之间的平方误差误差
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

myMat =mat(loadDataSet('exp2.txt'))
print  createTree(myMat,modelLeaf,modelErr,(1,10))



'''用树回归进行预测的代码'''

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)
#自顶向下遍历整棵树，直到命中叶节点为止。
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
    
    
trainMat=mat(loadDataSet('bikeSpeedVsIq_train.txt'))
testMat=mat(loadDataSet('bikeSpeedVsIq_test.txt'))
#回归树
myTree = createTree(trainMat,ops=(1,20))
yHat = createForeCast(myTree,testMat[:,0])
print corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]     #0.964085231822

#模型树
myModelTree = createTree(trainMat,modelLeaf,modelErr,(1,20))
yHat = createForeCast(myModelTree,testMat[:,0],modelTreeEval)
print corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]     #0.976041219138

#标准的线性回归
ws,X,Y =linearSolve(trainMat)
for i in range(shape(testMat)[0]):
    yHat[i] = testMat[i,0] * ws[1,0] + ws[0,0]
print corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]     #0.943468423567
