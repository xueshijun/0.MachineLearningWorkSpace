# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Wed Feb 24 23:47:58 2016
#Version: 1.0
#-------------------------------------------------------------------------------
'''
降维技术使得数据变得更易使用， 并且它们往往能够去除数据中的噪声， 使得其他机器学习
任务更加精确。降维往往作为预处理步骤， 在数据应用到其他算法之前清洗数据。

降维技术:
    1)主成分分析(Principal Component Analysis, PCA)
        PCA可以从数据中识别其主要特征，它是通过沿着数据最大方差方向旋转坐标轴来实现的。
        选择方差最大的方向作为第一条坐标轴，后续坐标轴则与前面的坐标轴正交。
        协方差矩阵上的特征值分析可以用一系列的正交坐标轴来获取。

        在PCA中，数据从原来的坐标系转换到了新的坐标系，新坐标系的选择是由数据本身决定的。
        第一个新坐标轴选择的是原始数据中方差最大的方向，第二个新坐标轴的选择和第一个坐标轴正交且具有最大方差的方向。
        该过程一直重复，重复次数为原始数据中特征的数目。
        
        我们会发现，大部分方差都包含在最前面的几个新坐标轴中。因此，我们可以忽略余下的坐标轴，即对数据进行了降维处理。
        注:第一条坐标轴/数据最大方差方向:覆盖数据最大差异性的坐标轴
           第二条坐标轴/数据次大差异性的坐标轴:覆盖数据次大差异性的坐标轴
           严谨的说法就是正交:当然，在二维平面下，垂直和正交是一回事。
    2)降维技术是因子分析(Factor Analysis)
        在因子分析中，我们假设在观察数据的生成中有一些观察不到的隐变量(latent variable)。
        假设观察数据是这些隐变量和某些噪声的线性组合。
        那么隐变量的数据可能比观察数据的数目少，也就是说通过找到隐变量就可以实现数据的降维。
        因子分析已经应用于社会科学、金融和其他领域中了。
    3)独立成分分析(Independent Component Analysis,ICA)
        ICA假设数据是从N个数据源生成的，这一点和因子分析有些类似。
        假设数据为多个数据源的混合观察结果,这些数据源之间在统计上是相互独立的，而在PCA人中只假设数据是不相关的。
        同因子分析一样,如果数据源的数目少于观察数据的数目，则可以实现降维过程。
    4)奇异值分解(SVD)

将所有的数据集都调人了内存,如果无法做到,就需要其他的方法来寻找其特征值。
如果使用在线PCA分析的方法，可参考"Incremental Eigenanalysis for Classification"

主成分分析
    优点：降低数据的复杂性，识别最重要的多个特征。
    缺点：不一定需要, 且可能损失有用信息。
    适用数据类型：数值型数据。
    

将数据转换成前N个主成分的伪码大致如下：
去除平均值
计算协方差矩阵
计算协方差矩阵的特征值和特征向量
将特征值从大到小排序
保留最上面的N个特征向量
将数据转换到上述N个特征向量构建的新空间中
'''
from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):#dataMat:进行PCA操作的数据集,topNfeat:应用的N个特征

    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals    #首先计算并减去原始数据集的平均值
    covMat = cov(meanRemoved, rowvar=0) #计算协方差矩阵及其特征值
    eigVals,eigVects = linalg.eig(mat(covMat))
    
    eigValInd = argsort(eigVals)        #对特征值进行从小到大的排序。
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #去掉不想要的dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects     #transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


dataMat = loadDataSet('testSet.txt')
lowDMat,reconMat = pca(dataMat,1)
print shape(lowDMat) #(1000,1)
 
def drawFirstPrincipalComponent(datMat,recoMat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datMat[:,0].flatten().A[0],datMat[:,1].flatten().A[0],marker='^',s=90)
    ax.scatter(recoMat[:,0].flatten().A[0],recoMat[:,1].flatten().A[0],marker='o',s=50,c='red')
    plt.show()
drawFirstPrincipalComponent(dataMat,reconMat)
'''
示例:利用PCA对半导体制造数据降维
该数据包含很多的缺失值。这些缺失值是以NaN(Not a Number的缩写)标识的。


在590个特征下，几乎所有样本都有NaN,因此去除不完整的样本不太现实。
尽管我们可以将所有的NaN替换成0,但是由于并不知道这些值的意义,所以这样做是个下策。
下面我们。

'''

def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        #用平均值来代替缺失值，平均值根据那些非NaN得到        
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat

def draw():
    import matplotlib.pyplot as plt
    #below is a quick hack copied from pca.pca()
    dataMat = replaceNanWithMean()
    #去除均值
    meanRemoved = dataMat - mean(dataMat,axis=0)
    covMat = cov(meanRemoved,rowvar=0)  #计算协方差矩阵
    eigVals,eigVects=linalg.eig(mat(covMat)) #对该矩阵进行特征值分析
    
#==============================================================================
#     i=0;
#     j=0;
#     for eigVal in eigVals:
#         if eigVal>0:i=i+1
#         else:j=j+1
#     print i,j
#==============================================================================
    x = np.array([1,2,3,5,7,4,3,2,8,0])
    mask = x<5
    mx = ma.array(x,mask=mask)
    print len(mx)
    mask = eigVals < 0
    mx = ma.array(eigVals,mask=mask)
    
    
    eigValInd = argsort(eigVals)   #sort, sort goes smallest to largest
    eigValInd = eigValInd[::-1]#reverse
    sortedEigVals = eigVals[eigValInd]
    total = sum(sortedEigVals)
    varPercentage = sortedEigVals/total*100
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 21), varPercentage[:20], marker='^')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Percentage of Variance')
    plt.show()
draw()