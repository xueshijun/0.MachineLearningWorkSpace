# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Tue Feb 16 10:14:32 2016
#Version: 1.0
#-------------------------------------------------------------------------------
'''示例:在一个难数据集上的AdaBoost的应用
(1)收集数据：提供的文本文件。
(2)准备数据：确保类别标签是+1和-1而非1和0。
(3)分析数据：手工检查数据。
(4)训练算法：在数据上，利用adaBoostTrainsDS()函数训练出一系列的分类器。
(5)测试算法：我们拥有两个数据集。在不釆用随机抽样的方法下，我们就会对AdaBoost和Logistic回归的结果进行完全对等的比较。
(6)使用算法：观察该例子上的错误率。不过，也可以构建一个Web网站，让驯马师输入马的症状然后预测马是否会死去。
'''
import AdaBoost

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

datArr,labelArr = loadDataSet('horseColicTest2.txt')
weakClassArr,aggClassEst = AdaBoost.adaBoostTrainDS(datArr,labelArr,40)
'''
testDataArr,testLabelArr= loadDataSet('horseColicTest2.txt')
prediction10=AdaBoost.adaClassify(testDataArr,weakClassArr)
errArr = mat(ones((67,1)))
#统计错误数（错误率=错误数/67）
errArr[prediction10!=mat(testLabelArr).T].sum()
'''


'''非均衡分类问题
假设所有类别的分类代价,在大多数情况下不同类别的分类代价并不相等。
1.调节分类器的阈值
    一种不同分类器的评价方法:ROC曲线、AUC
    度量分类器性能的指标:构建一个同时使正确率和召回率最大的分类器是具有挑战性的。
'''
import AdaBoostROC as ad
print '***************************'
print aggClassEst,labelArr
ad.plotROC(aggClassEst,labelArr)

'''
2.基于代价函数的分类器决策控制
代价敏感的学习(cost-sensitivelearning)
    -------------------------------------------------------------
    分类器的代价矩阵(代价不是0就是1)
                          预测结果
                        +1      -1
                +1      0        1  
    真实结果   
                -1      1        0
                
    基于该代价矩阵计算其总代价：TP*0 + FN*1 + FP*1 + TN*0
    -------------------------------------------------------------
                          预测结果
                        +1      -1
                +1      -5        1  
    真实结果   
                -1      50        0   
    
    基于该代价矩阵计算其总代价：TP*(-5)+FN*l+FP*50+TN*0
    -------------------------------------------------------------
    在分类算法中,我们有很多方法可以用来引人代价信息。
    在AdaBoost中,可以基于代价函数来调整错误权重向量D。
    在朴素贝叶斯中,可以选择具有最小期望代价而不是最大概率的类别作为最后的结果。
    在SVM中,可以在代价函数中对于不同的类别选择不同的参数0。
    上述做法就会给较小类更多的权重，即在训练时，小类当中只允许更少的错误。
3.处理非均衡问题的数据抽样方法
欠抽样(undersampling):意味着删除样例。
过抽样(oversampling):意味着复制样例

信用卡欺诈中
    正例少,反例多：
        保留正例类别中的所有样例(正例类别属于罕见类别)，而对反例类别进行欠抽样或者样例删除处理
        缺点:要确定哪些样例需要进行删除。但是,在选择删除的样例中可能携带了剩余样例中并不包含的有价值信息。
        解决办法:择那些离决策边界较远的样例进行删除
    正例多,反例少
        对正例进行欠抽样处理
        缺点:极端
        解决方案:使用反例类别的欠抽样和正例类别的过抽样相混合的方法
            对正例类别进行过抽样， 我们可以复制已有样例或者加人与已有样例相似的点。
            一种方法是加人已有数据点的插值点，但是这种做法可能会导致过拟合的问题。
'''

'''