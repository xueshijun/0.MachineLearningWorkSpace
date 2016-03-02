# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Sun Feb 14 23:25:15 2016
#Version: 1.0
#-------------------------------------------------------------------------------
'''
集成方法(ensemble method)或者元算法(meta-algorithm)是将不同的分类器组合起来。

使用集成方法时会有多种形式:
    可以是不同算法的集成,
    也可以是同一算法在不同设置下的集成,
    还可以是数据集不同部分分配给不同分类器之后的集成。

AdaBoost
优点：泛化错误率低，易编码，可以应用在大部分分类器上，无参数调整。
缺点：对离群点敏感。
适用数据类型：数值型和标称型数据。


--------------------------------------
bagging：基于数据随机重抽样的分类器构建方法
自举汇聚法(bootstrap aggregating)，也称为bagging方法,是在从原始数据集选择S次后得到S个新数据集的一种技术。
新数据集和原数据集的大小相等。每个数据集都是通过在原始数据集中随机选择一个样本来进行替换而得到的。
这里的替换就意味着可以多次地选择同一样本。这一性质就允许新数据集中可以有重复的值，而原始数据集的某些值在新集合中则不再出现。

在S个数据集建好之后，将某个学习算法分别作用于每个数据集就得到了S个分类器。当我们要对新数据进行分类时，就可以应用这S个分类器进行分类。
与此同时，选择分类器投票结果中最多的类别作为最后的分类结果。

-------------
更先进的bagging方法，比如随机森林(random forest)有关这些方法的一个很好的讨论材料
参见网页http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm。


---------------
两种集成方法是boosting、bagging、随机森林
boosting其中一个最流行的版本AdaBoost(adaptive boosting,自适应boosting).



boosting vs bagging
相同：不论是在boosting还是bagging当中，所使用的多个分类器的类型都是一致的。
不同： 1)在bagging中，是通过随机抽样的替换方式，得到了与原始数据集规模一样的数据集。
        boosting中，在数据集上顺序应用了多个不同的分类器，每个新分类器都根据已训练出的分类器的性能来进行训练。
        boosting是通过集中关注被已有分类器错分的那些数据来获得新的分类器。
       2)boosting分类的结果是基于所有分类器的加权求和结果的,分类器权重并不相等，每个权重代表的是其对应分类器在上一轮迭代中的成功度。
       bagging中的分类器权重是相等的


AdaBoost
以弱分类器作为基分类器，并且输人数据，使其通过权重向量进行加权。在第一次迭代当中，所有数据都等权重。
但是在后续的迭代当中，前次迭代中分错的数据的权重会增大。

AdaBoost理论基础：
    使用弱分类器和多个实例来构建一个强分类器
    （“弱”意味着分类器的性能比随机猜测要略好,但是也不会好太多。
    就是说，在二分类情况下弱分类器的错误率会高于50%,而"强"分类器的错误率将会低很多)
    
AdaBoosting运行过程:
    训练数据中的每个样本，并赋予其一个权重，这些权重构成了向量D。
    一开始，这些权重都初始化成相等值。
    首先在训练数据上训练出一个弱分类器并计算该分类器的错误率，然后在同一数据集上再次训练弱分类器。
    在分类器的第二次训练当中，将会重新调整每个样本的权重，其中第一次分对的样本的权重将会降低，而第一次分错的样本的权重将会提高。
    为了从所有弱分类器中得到最终的分类结果，AdaBoost为每个分类器都分配了一个权重值alpha，这些alpha值是基于每个弱分类器的错误率进行计算的。
        其中，错误率的定义为：
             alpha的计算公式:   
    计算出alpha值之后，可以对权重向量D进行更新，以使得那些正确分类的样本的权重降低而错分样本的权重升高。
        D的计算方法如下:
            如果某个样本被正确分类，那么该样本的权重更改为:
            如果某个样本被错分，那么该样本的权重更改为:


AdaBoost的一般流程
(1)收集数据：可以使用任意方法。
(2)准备数据：依赖于所使用的弱分类器类型，本章使用的是单层决策树，这种分类器可以处理任何数据类型。
            当然也可以使用任意分类器作为弱分类器，第2章到第6章中的任一分类器都可以充当弱分类器。
            作为弱分类器，简单分类器的效果更好。
(3)分析数据：可以使用任意方法。
(4)训练算法：AdaBoost的大部分时间都用在训练上，分类器将多次在同一数据集上训练弱分类器。
(5)测试算法：计算分类的错误率
(6)使用算法：同SVM一样,AdaBoost预测两个类别中的一个。如果想把它应用到多个类别的场合，那么就要像多类别SVM中的做法一样对AdaBoost进行修改。
'''


from numpy import *
def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
                    [ 2. ,  1.1],
                    [ 1.3,  1. ],
                    [ 1. ,  1. ],
                    [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

datMat,classLabels=loadSimpData()

#========================================================================
'''弱分类器之单层决策树(decision stump,也称决策树桩),仅基于单个特征来做决策。
'''
'''通过阈值比较对数据进行了分类'''
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    
'''它会在一个加权数据集中循环，并找到具有最低错误率的单层决策树。伪代码如下:
将最小错误率minError设为+00
对数据集中的每一个特征(第一层循环)：
    对每个步长(第二层循环)：
        对每个不等号(第三层循环)：
            建立一棵单层决策树并利用加权数据集对它进行测试
            如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
返回最佳单层决策树
'''
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr);  labelMat = mat(classLabels).T;   m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

#D是一个概率分布向量，因此其所有的元素之和为1.0。一开始的所有元素都会被初始化成1/m
D = mat(ones((5,1))/5)

bestStump,minError,bestClasEst=buildStump(datMat,classLabels,D)


'''基于单层决策树的AdaBoost训练过程,伪代码如下:

对每次迭代：
    利用buildStump()函数找到最佳的单层决策树
    将最佳单层决策树加入到单层决策树数组
    计算alpha
    计算新的权重向量D
    更新累计类别估计值
    如果错误率等于0.0,则退出循环
'''
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        print "classEst: ",classEst.T
        #为下一次迭代计算D
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        
        #错误率累加计算
        aggClassEst += alpha*classEst
        print "aggClassEst: ",aggClassEst.T        
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate
        
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst
weakClassArr,aggClassEst=adaBoostTrainDS(datMat,classLabels,9)
 
'''每个弱分类器的结果以其对应的alpha值作为权重。所有这些弱分类器的结果加权求和就得到了最后的结果。
datToClass：由一个或者多个待分类样例
classifierArr：多个弱分类器组成的数组
predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#c
'''
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)): 
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)

'''随着迭代的进行，数据点[0,0]的分类结果越来越强'''
print '----------------------------'
print adaClassify([0,0],weakClassArr)
print '----------------------------'
print adaClassify([[5,5],[0,0]],weakClassArr)