# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Mon Feb 08 13:28:03 2016
#Version: 1.0
#-------------------------------------------------------------------------------
'''基于最大间隔分隔数据
优点 ：泛化错误率低，计算开销不大，结果易解释。
缺点 ：对参数调节和核函数的选择敏感，原始分类器不加修改仅适用于处理二类问题。
适用数据类型：数值型和标称型数据。


        SVM的一般流程
(1)收集数据：可以使用任意方法。
(2)准备数据：需要数值型数据。
(3)分析数据：有助于可视化分隔超平面。
(4)训练算法：SVM的大部分时间都源自训练，该过程主要实现两个参数的调优。
(5)测试算法：十分简单的计算过程就可以实现。
(6)使用算法：几乎所有分类问题都可以使用SVM，值得一提的是，SVM本身是一个二类分类器，对多类问题应用SVM需要对代码做一些修改。
'''
from numpy import *
def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineAttr = line.strip().split('\t')
        dataMat.append([float(lineAttr[0]),float(lineAttr[1])])
        labelMat.append(float(lineAttr[2]))
    return dataMat,labelMat

dataAttr,labelAttr = loadDataSet('testSet.txt')
#print labelAttr

'''
SMO:序列最小优化(Sequential Minimal Optimization)

SMO算法的目标:求出一系列alpha和b,一旦求出了这些alpha, 就很容易计算出权重向量w并得到分隔超平面。
SMO算法的工作原理：每次循环中选择两个alpha进行优化处理。一旦找到一对合适的alpha,那么就增大其中一个同时减小另一个。
    "合适"就是指两个alpha必须要符合一定的条件:
        1.这两个alpha必须要在间隔边界之外
        2.这两个alpha还没有进行过区间化处理或者不在边界上。
        
        
SMO函数的伪代码
创建一个alpha向量并将其初始化为O向量
当迭代次数小于最大迭代次数时(外循环):
    对数据集中的每个数据向量（内循环)：
        如果该数据向量可以被优化：
            随机选择另外一个数据向量
            同时优化这两个向量
            如果两个向量都不能被优化，退出内循环
    如果所有向量都没被优化，增加迭代数目，继续下一次循环
'''


def selectJrand(i,m):
    j = i
    while(j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H :aj = H;
    if L >aj:aj = L;
    return aj
    
'''
简化版SMO算法的运行是没有什么问题的， 但是在更大的数据集上的运行速度就会变慢。
5个输人参数:数据集、类别标签、常数0 、容错率和取消前最大的循环次数'''
def SOMSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = mat(dataMatIn);
    labelMat=mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            #如果alpha可以更改,则进入优化过程
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)#随机选择第二个alpha
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); 
                alphaJold = alphas[j].copy();
                #保证alpha在0与C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                
                if L==H: print "L==H"; continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                #对i进行修改,修改量与j相同,但方向相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                #设置常数项
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas

            
#b,alphas = SOMSimple(dataAttr,labelAttr,0.6,0.001,40) 

'''======================================================================================================='''
'''
在这两个版本中，实现alpha的更改和代数运算的优化环节一模一样。
在优化过程中，唯一的不同就是选择alpha的方式。
完整版的Platt SMO算法应用了一些能够提速的启发方法。

Platt SMO算法是通过一个外循环来选择第一个alpha值的，并且其选择过程会在两种方式之间进行交替： 
一种方式是在所有数据集上进行单遍扫描， 另一种方式则是在非边界alpha(非边界alpha指的就是那些不等于边界0或C的alpha值)中实现单遍扫描。


'''

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #误差缓存
        self.K = mat(zeros((self.m,self.m))) 
'''内循环中的启发式方法，计算E值并返回'''
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
'''选择第二个alpha或者说内循环的alpha值'''       
def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):#选择具有最大步长的j
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
            
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #第二个alpha选择中的启发式方法
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

''' 完整版Platt SMO的外循环代码'''
def SOMComplete(dataMatIn, classLabels,C,toler,maxIter, kTup = ('lin', 0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs: 
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print "iteration number: %d" % iter
    return oS.b,oS.alphas

#b,alphas = SOMComplete(dataAttr,labelAttr,0.6,0.001,40)


def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

#ws = calcWs(alphas,dataAttr,labelAttr)
#print ws

'''在对数据进行分类处理，比如对说第一个数据点分类，可以这样输入：
'''

dataMat = mat(dataAttr)
''' 如果该值大于0，那么其属于1类;
    如果该值小于0，那么则属于-1类。
    对于数据点0 ，则属于-1类。'''
print dataMat[0] * mat(ws) + b 
print labelAttr[0]

print dataMat[1] * mat(ws) + b 
print labelAttr[1]
  