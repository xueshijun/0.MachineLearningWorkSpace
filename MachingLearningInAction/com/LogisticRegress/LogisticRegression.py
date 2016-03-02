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
优点：计算代价不高，易于理解和实现。
缺点：容易欠拟合，分类精度可能不高。
适用数据类型：数值型和标称型数据。
Logistic回归的目的是寻找一个非线性函数Sigmoid的最佳拟合参数，求解过程可以由最优化算法来完成。
在最优化算法中，最常用的就是梯度上升算法，而梯度上升算法又可以简化为随机梯度上升算法。
随机梯度上升算法与梯度上升算法的效果相当，但占用更少的计算资源。
此外，随机梯度上升是一个在线算法，它可以在新数据到来时就完成参数更新，而不需要重新读取整个数据集来进行批处理运算。

机器学习的一个重要问题就是如何处理缺失数据。这个问题没有标准答案，取决于实际应用中的需求。
现有一些解决方案，每种方案都各有优缺点。

Logistic回归的一般过程
(1)收集数据：采用任意方法收集数据。
(2)准备数据：由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳。
(3)分析数据：采用任意方法对数据进行分析。
(4)训练算法：大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数。
(5)测试算法：一旦训练步驟完成，分类将会很快。
(6)使用算法：首先，我们需要输入一些数据，并将其转换成对应的结构化数值；
接着，基于训练好的回归系数就可以对这些数值进行简单的回归计算，判定它们属于哪个类别. 
在这之后，我们就可以夺输出的类别上做一些其他分析工作。
'''
def loadDataSet():
    dataMat = []
    labelMat=[]
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(intX):
    return  1.0 /(1+exp(-intX))
    

'''画出数据集和Logistic回归最佳拟合直线的函数'''
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1] * x ) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
'''1.梯度上升算法
梯度上升算法在每次更新回归系数时都需要遍历整个数据集, 
该方法在处理100个左右的数据集时尚可，但如果有数十亿样本和成千上万的特征，那么该方法的计算复杂度就太高了。

    

一种改进方法是一次仅用一个样本点来更新回归系数， 该方法称为随机梯度上升算法。
由于可以在新样本到来时对分类器进行增量式更新，因而随机梯度上升算法是一个在线学习算法。
与"在线学习"相对应，一次处理所有数据被称作是"批处理"。
'''
def gradAscent(dataMatIn,classLabels):
    '''转换为NumPy矩阵数据类型'''
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    
    m,n = shape(dataMatrix)
    alpha = 0.001       #向目标移动的步长，：
    maxCycles = 500     #迭代次数
    weights = ones((n,1))
    '''矩阵相乘'''
    for k in range(maxCycles):
        h=sigmoid(dataMatrix * weights)
        error =(labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights

dataAttr,labelMat = loadDataSet()
weights = gradAscent(dataAttr,labelMat)

#plotBestFit(weights)
'''2.训练算法：随机梯度上升
系数达到稳定值的迭代次数差距大：例如x2只经过50次迭代就达到了稳定值，而x1和x0则需要更多次的迭代。
另外，在大的波动停止后，还有一些小的周期性波动.
现象的原因是:存在一些不能正确分类的样本点（数据集并非线性可分)，在每次迭代时会引发系数的剧烈改变。
伪代码：
每个回归系数初始化为1
重复R次:
    计算整个数据集的梯度
    使用alpha * gradient更新回归系数的向量
返回回归系数
    
    
我们期望算法能避免来回波动，从而收敛到某个值。另外，收敛速度也需要加快。
看算法3
'''
def stocGranAscent(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)                           #所有回归系数初始化为1
    for i in range(m):                          #对数据集中每个样本
        h = sigmoid(sum(dataMatrix[i]*weights)) #计算该样本的梯度
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i] #使用alpha x gradient更新回归系数值
    return weights                              #返回回归系数值

weights = stocGranAscent(array(dataAttr),labelMat)
#plotBestFit(weights)
'''
1，2梯度上升算法一些区别：
第一，后者的变量h和误差error 都是向量，而前者则全是数值；
第二，前者没有矩阵的转换过程，所有变量的数据类型都是NumPy数组。
'''

'''3.改进的随机梯度上升算法'''
def stocGranAscentImprove(dataMatrix,classLabels,numIter = 150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            '''alpha每次迭代时需要调整,这会缓解系数的波动或者高频波动
            虽然alpha会随着迭代次数不断减小，但永远不会减小到0，因为还存在一个常数项。
            
            必须这样做的原因是为了保证在多次迭代之后新数据仍然具有一定的影响。
            如果要处理的问题是动态变化的，那么可以适当加大上述常数项，来确保新的值获得更大的回归系数。
            
            另一点值得注意的是，在降低alpha的函数中，alpha每次减少l/(j+i)，其中j是迭代次数，i是样本点的下标。
            这样当j<<max(i)时 ，alpha就不是严格下降的。
            避免参数的严格下降也常见于模拟退火算法等其他优化算法中。
            '''
            alpha = 4 / (1.0+j+i) + 0.0001
            
            '''通过随机选取样本来更新回归系数。
            这种方法将减少周期性的波动。这种方法每次随机从列表中选出一个值，然后从列表中删掉该值(再进行下一次迭代)'''
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
