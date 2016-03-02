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
使用朴素贝叶斯分类器从个人广告中获取区域倾向

我们将分别从美国的两个城市中选取一些人，通过分析这些人发布的征婚广告信息，来比较这两个城市的人们在广告用词上是否不同。

(1)收集数据：从RSS源收集内容，这里需要对RSS源构建一个接口。
(2)准备数据：将文本文件解析成词条向量。
(3)分析数据：检查词条确保解析的正确性。
(4)训练算法：使用我们之前建立的trainNB()函数。
(5)测试算法：观察错误率，确保分类器可用。可以修改切分程序，以降低错误率，提高分类结果。
(6)使用算法：构建一个完整的程序，封装所有内容。给定两个RSS源，该程序会显示最常用的公共词。
'''

#收集数据：导入RSS源
'''
Universal Feed Parser是Python中最常用的RSS程序库。
https://github.com/kurtmckee/feedparser
python setup.py install
'''

def textParse(bigString):
    import re
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(bigString)      
    return [token.lower() for token in listOfTokens if len(token)>2]

def calcMostFreq(vocabList,fullText):
    '''计算出现频率'''
    import operator
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import bayes
    import numpy as np
    docList =[];classList = [];fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    '''每次访问一条RSS源'''
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)


        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    '''去掉出现次数最高的那些词
    
    另一个常用的方法是不仅移除高频词，同时从某个预定词表中移除结构上的辅助词。
    该词表称为停用词表(stop word list),目前可以找到许多停用词表
    （在本书写作期间， http://www.ranks.nL/resources/stopwords.html上有一个很好的多语言停用词列表）。
'''
    vocabList = bayes.createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:vocabList.remove(pairW[0])

    trainingSet = range(2*minLen);    
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bayes.bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam =bayes.trainNB(np.array(trainMat),np.array(trainClasses))
    
    errorCount = 0
    for docIndex in testSet:
        wordVector = bayes.bagOfWords2VecMN(vocabList,docList[docIndex])
        if bayes.classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount +=1
    print 'the error rate is :',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V



'''
RSS源要在函数外导人，这样做的原因是RSS源会随时间而改变。如果想通过改变代码来比较程序执行的差异，就应该使用相同的输入 。
重新加载RSS源就会得到新的数据，但很难确定是代码原因还是输人原因导致输出结果的改变。
'''
import feedparser
ny = feedparser.parse('http://newyork.craigslist.org/search/stp?format=rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
#print ny['entries']
#print ny['entries'][2]['summary']
#print len(ny['entries'])
vocabList,pSF,pNY = localWords(ny,sf)

#显示地域相关的用词
#最具表征性的词汇显示函数
#这些元组会按照它们的条件概率进行排序
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[];    topSF=[]
    for i in range(len(p0V)):
        
        #与之前返回排名最高的X个单词不同，这里可以返回大于某个阈值的所有词。
        if p0V[i] > -6.0:topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0:topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print 'SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF'
    print sortedSF
    '''for item in sortedSF:
        print item[0]'''
    
    sortedNY=sorted(topNY,key=lambda pair:pair[1],reverse=True)
    print 'NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY'
    print sortedNY
    '''for item in sortedNY:
        print item[0]'''

getTopWords(ny,sf)



