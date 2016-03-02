# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Wed Feb 17 15:01:29 2016
#Version: 1.0
#-------------------------------------------------------------------------------
from Regression import *
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()
abX,abY=loadDataSet('abalone.txt')
'''分析误差大小
可以看到 ,使用较小的核将得到较低的误差。那么 ,为什么不在所有数据集上都使用最小的核呢？
这是因为使用最小的核将造成过拟合，对新数据不一定能达到最好的预测效果。
'''
yHat01 = lwlrTest(abX[0:99],[abX[0:99],abY[0:99]],0.1)
yHat1  = lwlrTest(abX[0:99],[abX[0:99],abY[0:99]],1)
yHat10 = lwlrTest(abX[0:99],[abX[0:99],abY[0:99]],10)
print rssError(abY[0:99],yHat01.T)  #56.7881737401
print rssError(abY[0:99],yHat1.T)   #429.89056187
print rssError(abY[0:99],yHat10.T)  #549.118170882
#新数据集
yHat01 = lwlrTest(abX[100:199],[abX[100:199],abY[100:199]],0.1)
yHat1  = lwlrTest(abX[100:199],[abX[100:199],abY[100:199]],1)
yHat10 = lwlrTest(abX[100:199],[abX[100:199],abY[100:199]],10) 
print rssError(abY[100:199],yHat01.T)  #3309.75655115
print rssError(abY[100:199],yHat1.T)   #231.813447969
print rssError(abY[100:199],yHat10.T)  #291.879963906

#简单线性回归达到了与局部加权线性回归类似的效果
ws=standRegres(abX[100:199],abY[100:199])
yHat = mat(abX[100:199]) * ws
print rssError(abY[100:199],yHat.T.A)  #292.70429869

