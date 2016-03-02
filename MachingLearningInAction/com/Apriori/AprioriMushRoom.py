# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Tue Feb 23 11:08:20 2016
#Version: 1.0
#-------------------------------------------------------------------------------
'''mushroom.dat'''
import Apriori  as ap
mushDataSet = [line.split() for line in open('mushroom.dat').readlines()]
L,supportData = ap.apriori(mushDataSet,minSupport=0.3)

for item in L[1]:
    if item.intersection('2'):print item

for item in L[3]:
    if item.intersection('2'):print item