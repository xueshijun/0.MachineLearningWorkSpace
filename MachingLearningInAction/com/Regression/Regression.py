# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Wed Feb 17 09:36:59 2016
#Version: 1.0
#-------------------------------------------------------------------------------
'''用线性回归找到最佳拟合直线

线性回归
优 点 ：结果易于理解，计算上不复杂。
缺 点 ：对非线性的数据拟合不好。
适用数据类型：数值型和标称型数据。

回归的目的是预测数值型的目标值。
求解回归方程(regression equation)的回归系数(regression weights)的过程即回归。

回归分为线性回归和非线性回归,一般都是指线性回归（ linearregression)


回归的一般方法
(1)收集数据：采用任意方法收集数据。
(2)准备数据：回归需要数值型数据，标称型数据将被转成二值型数据。
(3)分析数据：绘出数据的可视化二维图将有助于对数据做出理解和分析，
    在采用缩减法求得新回归系数之后, 可以将新拟合线绘在图上作为对比。
(4)训练算法：找到回归系数。
(5)测试算法：使用幻或者预测值和数据的拟合度，来分析模型的效果。
(6)使用算法：使用回归，可以在给定输入的时候预测出一个数值，这是对分类方法的提升 ，
    因为这样可以预测连续型数据而不仅仅是离散的类别标签。
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t') 
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr);  
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
'''用来计算最佳拟合直线
Numpy提供一个线性代数的库linalg,其中包含很多有用的函数。
    可以直接调用linalg.det()来计算行列式。
Numpy的线性代数库还提供一个函数来解未知矩阵，
    ws=linalg.solve(xTx,xMat.T*yMatT)
'''
def standRegres(datMat,labelMat):
    xMat = mat(datMat); yMat = mat(labelMat).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:#如果没有检查行列式是否为零就试图计算矩阵的逆，将会出现错误。
        print "This matrix is singular, cannot do inverse"
        return 
    ws = xTx.I * (xMat.T*yMat)
    return ws

dataMat,labelMat = loadDataSet('ex0.txt') 
ws = standRegres(dataMat,labelMat)
print ws

'''
预测
yHat=xMat*ws
'''
'''绘出数据集散点图和最佳拟合直线图：'''
def drawImg(dataMat,labelMat):
    import matplotlib.pyplot as plt
    #创建了图像并绘出了原始的数据。 
    fig = plt.figure()
    ax =fig.add_subplot(111)
    ax.scatter(dataMat[:,1].flatten().A[0],labelMat.T[:,0].flatten().A[0])
    #绘制计算出的最佳拟合直线
    xCopy=dataMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1],yHat,'r')
    plt.show()  
    
xMat=mat(dataMat)
yMat=mat(labelMat)
drawImg(xMat,yMat)
'''
计算预测值yHat序列和真实值y序列的匹配程度，即计算这两个序列的相关系数。
'''
#保证两个向量都是行向量
yHat = xMat * ws
print "yHat和yMat相关系数:",corrcoef(yHat.T,yMat)

#==============================================================================
# 局部加权线性回归函数
#==============================================================================
'''
线性回归的一个问题是有可能出现欠拟合现象，因为它求的是具有最小均方误差的无偏估计.

局部加权线性回归(Locally Weighted Linear Regression, LWLR)
允许在估计中引人一些偏差，从而降低预测的均方误差。
使用局部加权线性回归来构建模型，可以得到比普通线性回归更好的效果。 
在该算法中,我们给待预测点附近的每个点赋予一定的权重；在这个子集上基于最小均方差来进行普通的回归。

局部加权线性回归的问题在于，每次必须在整个数据集上运行。也就是说为了做出预测，必须保存所有的训练数据。
'''
def lwlr(testPoint,trainSet,k=1.0):
    xArr,yArr=trainSet
    xMat = mat(xArr); yMat = mat(yArr).T
    m,n = xMat.shape ;weights = mat(eye((m))) #只含对角元素的权重矩阵
    for i in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[i,:]     #
        #根据高斯核,点testPoint与x(i)越近,w(i,i)将会越大。
        weights[i,i] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws
'''
k:输人参数欠控制衰减的速度
当k=0. 5时，大部分的数据都用于训练回归模型；
当k=0.01时，仅有很少的局部点被用于训练回归模型
'''
print lwlr(dataMat[0],[dataMat,labelMat],1.0)     #[[ 3.12204471]]
print lwlr(dataMat[0],[dataMat,labelMat],0.001)   #[[ 3.20175729]]

'''得到数据集里所有点的估计'''
def lwlrTest(testSet,trainSet,k=1.0):  #loops over all the data points and applies lwlr to each one
    xArr,yArr=trainSet    
    m = shape(xArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testSet[i],trainSet,k)
    return yHat
'''
当k=1.0时权重很大，如同将所有的数据视为等权重，得出的最佳拟合直线与标准的回归一致。
使用k=0.01得到了非常好的效果，抓住了数据的潜在模式。
k=0.003纳人了太多的噪声点，拟合的直线与数据点过于贴近。
'''
yHat=lwlrTest(mat(dataMat),[mat(dataMat),mat(labelMat)],0.01)



def drawImgLWLR(testSet,trainSet):#dataMat,labelMat,yHat
    import matplotlib.pyplot as plt
    xArr,yArr=trainSet
    #将数据点按序排序
    srtInd= xArr[:,1].argsort(0)
    xSort = xArr[srtInd][:,0,:]
    #创建了图像并绘出了原始的数据。 
    fig = plt.figure()
    ax =fig.add_subplot(111)
    #原始的数据
    ax.scatter(xArr[:,1].flatten().A[0],yArr.T.flatten().A[0],s=2,c='red')
    #
    ax.plot(xSort[:,1],testSet[srtInd])
    plt.show()
    
drawImgLWLR(yHat,[mat(dataMat),mat(labelMat)])

#==============================================================================
#第一种缩减系数方法：岭回归
#==============================================================================
'''
如果数据的特征比样本点还多
如果特征比样本点还多(n > m ),即输入数据的矩阵X不是满秩矩阵,即不能求逆。
'''

#计算回归系数实现了给定lambda下的岭回归求解(注:由于lambda是Python保留的关键字)
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    '''在普通回归方法可能会产生错误时，岭回归仍可以正常工作。
    那么是不是就不再需要检查行列式是否为零，对吗？    
    不完全对，如果lambda=0的时候一样可能会产生错误，所以这里仍需要做一个检查。'''
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xMat,yMat):
    #标准化处理：所有特征都减去各自的均值并除以方差

    yMean = mean(yMat,0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    '''这里的lambda应以指数级变化,这样可以看出lamda在取非常小的值时和取非常大的值时分别对结果造成的影响。'''
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

abX,abY=loadDataSet('abalone.txt')
xMat = mat(abX); yMat=mat(abY).T
ridgeWeights=ridgeTest(xMat,yMat)


def drawImgRidge():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()
drawImgRidge()

#==============================================================================
#第二种缩减系数方法：前向逐步回归
#==============================================================================
#数据标准化
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat
'''
它属于一种贪心算法，即每一步都尽可能减少误差。
一开始，所有的权重都设为1，然后每一步所做的决策是对某个权重增加或减少一个很小的值。
伪代码：
数据标准化，使其分布满足0均值和单位方差
在每轮迭代过程中：
    设置当前最小误差lowestError为正无穷
    对每个特征：
        增大或缩小：
            改变一个系数得到一个新的
            计算新时下的误差
            如果误差Error小于当前最小误差lowestError：设置Wbest等于当前的W
        将W设置为新的Wbest 
eps  :表示每次迭代需要调整的步长
numIt:表示迭代次数
'''
def stageWise(xArr,yArr,eps=0.01,numIter=100):
    xMat = mat(xArr); yMat=mat(yArr).T; 
    xMat = regularize(xMat)#特征按照均值为0方差为1进行标准化处理
    yMat = yMat - mean(yMat,0)     #can also regularize ys but will get smaller coef

    m,n=shape(xMat); returnMat = zeros((numIter,n))
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()#为了实现贪心算法建立了ws的两份副本。
    for i in range(numIter):
        print ws.T;  lowestError = inf; #设置当前最小误差lowestError为正无穷
        for j in range(n):#对每个特征增大或缩小
            for sign in [-1,1]:
                wsTest = ws.copy(); wsTest[j] += eps*sign#对每个特征增大或缩小：
                yTest = xMat*wsTest; rssE = ((yMat.A-yTest.A)**2).sum()#计算新时下的误差
                if rssE < lowestError:
                    lowestError = rssE;wsMax = wsTest #：设 置咖03七等于当前的柯
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

#一段时间后系数就已经饱和并在特定值之间来回震荡，这是因为步长太大的缘故
stageWise(abX,abY,0.01,200)
stageWise(abX,abY,0.001,5000)
stageWise(abX,abY,0.01,200)
