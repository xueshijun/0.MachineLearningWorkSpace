# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Sun Feb 28 08:41:00 2016
#Version: 1.0
#-------------------------------------------------------------------------------
'''
奇异值分解
优 点 ：简化数据，去除嗓声，提高算法的结果。
缺 点 ：数据的转换可能难以理解。
适用数据类型：数值型数据。

SVD应用:信息检索：
    利用SVD的方法为隐性语义索引(Latent Semantic Indexing, LSI) 或隐性语义分析(LatentSemanticAnalysis, LSA)。
    
    在SVD中，一个矩阵是由文档和词语组成的。当我们在该矩阵上应用SVD时,就会构建出多个奇异值。
    这些奇异值代表了文档中的概念或主题,这一特点可以用于更高效的文档搜索。
    在词语拼写错误时,只基于词语存在与否的简单搜索方法会遇到问题。
    简单搜索的另一个问题就是同义词的使用。这就是说，当我们查找一个词时,其同义词所在的文档可能并不会匹配上。
    如果我们从上千篇相似的文档中抽取出概念,那么同义词就会映射为同一概念。
SVD应用:推荐系统
    简单版本的推荐系统能够计算项或者人之间的相似度。
    更先进的方法则先利用SVD从数据中构建一个主题空间,然后再在该空间下计算其相似度。 
'''

from numpy import *
U,Sigma,VT=np.linalg.svd([[1,1],[1,7]])
print U
print Sigma #仅返回对角元素
print VT

def loadExData():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]

Data=mat(loadExData())
U,Sigma,VT=linalg.svd(Data)
#用如下结果来近似原始数据集
Sig3 = mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]])
print U[:,:3] * Sig3*VT[:3,:]

'''
协同过滤(collaborative filtering)是通过将用户和其他用户的数据进行对比来实现推荐的。

相似度计算:
因为描述食品的属性和描述餐具的属性有所不同,不利用这些计算它们之间的相似度,
而是利用用户对它们的意见来计算相似度。
它并不关心物品的描述属性，而是严格地按照许多用户的观点来计算相似度。
'''

def ecludSim(inA,inB):
    return 1.0/(1.0 + linalg.norm(inA - inB))
#该方法相对于欧氏距离的一个优势在于，它对用户评级的量级并不敏感。
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]
#如果夹角为90度,则相似度为0;如果两个向量的方向相同,则相似度为1.0。
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5+0.5*(num/denom)

myMat=mat(loadExData())
#==============================================================================
# ecludSim(myMat[:,],myMat[:,4])
# ecludSim(myMat[:,],myMat[:,0])
# 
# cosSim(myMat[:,],myMat[:,4])
# cosSim(myMat[:,],myMat[:,0])
#
# pearsSim(myMat[:,],myMat[:,4])
#==============================================================================


'''
基于物品的相似度还是基于用户的相似度？
    基于物品(item-based)的相似度
        计算的时间会随物品数量的增加而增加，
    
    基于用户(user-based)的相似度
        计算的时间则会随用户数量的增加而增加。
    
推荐引擎的评价
    既没有预测的目标值，也没有用户来调査他们对预测的满意程度。
    可以采用前面多次使用的交叉测试的方法:将某些已知的评分值去掉,然后对它们进行预测,最后计算预测值和真实值之间的差异。
推荐引擎评价的指标是称为最小均方根误差(RootMeanSquaredError,RMSE):
    如果评级在1星到5星这个范围内,而我们得到RMSE为1.0,那么就意味着我们的预测值和用户给出的真实评价相差了一个星级。
'''

'''推荐未尝过的菜肴
推荐系统的工作过程是：给定一个用户，系统会为此用户返回N个最好的推荐菜。
(1)寻找用户没有评级的菜肴，即在用户-物品矩阵中的0值；
(2)在用户没有评级的所有物品中，对每个物品预计一个可能的评级分数。
    这就是说，我们认为用户可能会对物品的打分(这就是相似度计算的初衷)；
(3)对这些物品的评分从高到低进行排序，返回前N个物品。

'''
#计算在给定相似度计算方法的条件下，用户对物品的估计评分值。
def standEst(dataMat, user, item, simMeas):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        #如果某个物品评分值为0 , 就意味着用户没有对该物品评分，跳过了这个物品。
        if userRating == 0: continue 
        #寻找两个用户都评级的物品,给岀的是两个物品当中已经被评分的那个元素
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        if len(overLap) == 0: similarity = 0
        else: #随后，相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积。
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    #通过除以所有的评分总和，对上述相似度评分的乘积进行归一化。
    else: return ratSimTotal/simTotal 
#产生了最高的n个推荐结果
#相似度计算方法和估计方法
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    #对给定的用户建立一个未评分的物品列表
    unratedItems = nonzero(dataMat[user,:].A == 0)[1]
    #如果不存在未评分物品，那么 就 退 出 函 数 ；
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    #否则，在所有的未评分物品上进行循环。
    for item in unratedItems:
        #对每个未评分物品,则通过调用standEst()来产生该物品的预测得分.
        #该物品的编号和估计得分值会放在一个元素列表estimatedScore 
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    #最后按照估计得分，对该列表进行排序并返回
    #该列表是从大到小逆序排列的，因此其第一个值就是最大值。
    return sorted(itemScores, key=lambda jj:jj[1], reverse=True)[:N]



def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


U,Sigma,VT=np.linalg.svd(mat(loadExData2()))
Sig2=Sigma**2
print sum(Sig2)
print sum(Sig2) * 0.9
print sum(Sig2[:2])
print sum(Sig2[:3])

#基于SVD的评分估计
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat) #该函数的不同之处就在于它在第3行对数据集进行SVD分解。
    Sig4 = mat(eye(4)*Sigma[:4]) #用这些奇异值构建出一个对角矩阵
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #用U矩阵将物品转换到低维空间中
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        #相似度计算是在低维空间下进行的
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        #对相似度求和,同时对相似度及对应评分值的乘积求
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal


#==============================================================================
# recommend(myMat,2,simMeas=ecludSim)
# #
# recommend(myMat,2,simMeas=pearsSim)
#==============================================================================

