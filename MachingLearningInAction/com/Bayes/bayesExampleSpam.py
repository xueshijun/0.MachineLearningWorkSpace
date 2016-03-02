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
示例：使用朴素贝叶斯对电子邮件进行分类
(1)收集数据：提供文本文件。
(2)准备数据：将文本文件解析成词条向量。
(3)分析数据：检查词条确保解析的正确性。
(4)训练算法：使用我们之前建立的trainNB()函数。
(5)测试算法：使用classifyNB()，并且构建一个新的测试函数来计算文档集的错误率。
(6)使用算法：构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上。
'''
from numpy import  *
import bayes
#测试算法：使用朴素贝叶斯进行交叉验证
def textParse(bigString):
    mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon'
    #mySent.split()      #标点符号也被当成了词的一部分
    #使用正则表示式来切分句子，其中分隔符是除单词、数字外的任意字符串。
    import re
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(mySent)      
    #0：空字符串
    #return [token.lower() for token in listOfTokens if len(token)>0]
    #2:文中包含en、py等字符，需要去掉
    return [token.lower() for token in listOfTokens if len(token)>2]

#贝叶斯垃圾邮件分类器进行自动化处理。
def spamTest():
    docList = []
    classList =[]
    fullText = []

    #导人文件夹spam与ham下的文本文件，并将它们解析为词列表
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        
    vocabList = bayes.createVocabList(docList)
    trainingSet = range(50);#本例中共有50封电子邮件,其中的值从0到49
    
    testSet = []    
    '''选择出的数字所对应的文档被添加到测试集， 同时也将其从训练集中剔除。    '''
    for i in range(10):#10封电子邮件被随机选择为测试集。
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMatrix=[];
    trainClasses = []
    for docIndex in trainingSet:
        trainMatrix.append(bayes.setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = bayes.trainNB(array(trainMatrix),array(trainClasses))
    
    errorCount=0
    for docIndex in testSet:
        #如果邮件分类错误，则错误数加1，最后给出总的错误百分比
        wordVector = bayes.setOfWords2Vec(vocabList,docList[docIndex])
        if bayes.classifyNB(array(wordVector),p0V,p1V,pSpam) !=classList[docIndex]:
            errorCount +=1
    print 'the error rate is :',float(errorCount) /len(testSet)
    
spamTest()
        
        
        
        
        
        
        
        