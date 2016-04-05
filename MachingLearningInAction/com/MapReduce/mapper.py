# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Thu Mar 03 09:26:08 2016
#Version: 1.0
#-------------------------------------------------------------------------------
import sys
from numpy import mat, mean, power
'''
该mapper首先按行读取所有的输人并创建一组对应的浮点数，然后得到数组的长度并创建NumPy矩阵。
再对所有的值进行平方，最后将均值和平方后的均值发送出去。这些值将用于计算全局的均值和方差

一个好的习惯是向标准错误输出发送报告。如果某作业10分钟内没有报告输出，则将Hadoop 中止。
'''
def read_input(file):
    for line in file:
        yield line.rstrip()
        
input = read_input(sys.stdin)#creates a list of input lines
input = [float(line) for line in input] #overwrite with floats
numInputs = len(input)
input = mat(input)
sqInput = power(input,2)

#output size, mean, mean(square values)
print "%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput)) #calc mean of columns
print >> sys.stderr, "report: still alive" 
