#coding=utf-8
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on 2015年8月30日
#Version: 1.0
#-------------------------------------------------------------------------------
'''
示例：使用决策树预测隐形眼镜类型
(1)收集数据：提供的文本文件。
(2)准备数据：解析tab键分隔的数据行。
(3)分析数据：快速检查数据，确保正确地解析数据内容
(4)训练算法：使用3.1节的createTree 函数。
(5)测试算法：编写测试函数验证决策树可以正确分类给定的数据实例。
(6)使用算法：存储树的数据结构，以便下次使用时无需重新构造树。
'''
import tree
fr=open('lenses.txt')

'''
#lenses=[inst.strip().split('\t') for inst in fr.readline()]
lensesLabels =['age','prescript','astigmatic','tearRate']
#lensesTree =  createTree(lenses,lensesLabels) 
'''
dataSet=[]
while True:
    line = fr.readline()
    if not line:break
    dataSet.append(line.strip().split('\t'))

lensesLabels =['age','prescript','astigmatic','tearRate']
lensesTree =  tree.createTree(dataSet,lensesLabels)
print lensesTree

'''
 匹配选项可能太多了。我们将这种问题称之为过度匹配(overfitting)。
 为了减少过度匹配问题，我们可以裁剪决策树，去掉一些不必要的叶子节点。
 如果叶子节点只能增加少许信息，则可以删除该节点，将它并人到其他叶子节点中。
'''