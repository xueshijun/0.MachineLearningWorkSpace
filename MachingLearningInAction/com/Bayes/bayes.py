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
优 点 ：在数据较少的情况下仍然有效，可以处理多类别问题。
缺 点 ：对于输入数据的准备方式较为敏感。
适用数据类型：标称型数据。

朴素贝叶斯的一般过程：
(1)收集数据：可以使用任何方法。本章使用尺88源。
(2)准备数据：需要数值型或者布尔型数据。
(3)分析数据：有大量特征时，绘制特征作用不大，此时使用直方图效果更好。
(4)训练算法：计算不同的独立特征的条件概率。
(5)测试算法：计算错误率。
(6)使用算法：一个常见的朴素贝叶斯应用是文档分类。可以在任意的分类场景中使用朴素贝叶斯命类器，不一定非要是文本。
'''
from numpy import  *
#词表到向暈的转换函数
# 创建了一些实验样本
def  loadDataSet():
    postingList  = [['my', 'dog', 'has', 'flea', 'problem'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit','buying', 'worthless', 'dog', 'food', 'stupid']]
    classVectors =[0,1,0,1,0,1]  #1代表侮辱性文字，0代表正常言论
    return postingList,classVectors


#单词集合
def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document) #两个集合的并集
    return list(vocabSet)
'''
#词集模型set-of-words model:将每个词的出现与否作为一个特征(每个词只能出现一次)
输人参数为词汇表及某个文档，输出的是文档向量，向量的每一元素为1或0，
分别表示词汇表中的单词在输人文档中是否出现 。
'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:#出现则为1
            returnVec[vocabList.index(word)] = 1
        else:
            returnVec[vocabList.index(word)] = 0    #可不写，默认为0
            print "the word: %s is not in my Vocabulary" % word
    return returnVec
'''
#词袋模型bag-of-words model:(每个单词可以出现多次)
'''
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
      
listPosts,listClasses = loadDataSet()
myVocabList= createVocabList(listPosts)

'''
计算每个类别中的文档数目
对每篇训练文档:
    对每个类别:
        如果词条出现文档中——>增加该词条的计数值
        增加所有词条的计数值
    对每个类别:
        对每个词条:
            将该词条的数目除以总词条数目得到条件概率
    返回每个类别的条件概率
'''
def trainNB(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) /float(numTrainDocs)
    #初始化程序中的分子变量和分母变量
    p0Num = zeros(numWords);            p1Num = zeros(numWords)
    #问题1
    p0Denom=0.0;                        p1Denom=0.0
    for i in range(numTrainDocs):#在for循环中，要遍历训练集trainMatrix中的所有文档。
        if trainCategory[i] == 1:   #一旦某个词语（侮辱性或正常词语）在某一文档中出现
            p1Num +=trainMatrix[i]      #则该词对应的个数（ PlNum或者p0Num) 就加1，
            p1Denom+= sum(trainMatrix[i]) #在所有的文档中，该文档的总词数也相应加1
        else:
            p0Num +=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    #问题2：
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect,p1Vect,pAbusive
    

'''
trainMatrix=[]          #构建了一个包含所有词的列表
for post in listPosts:  #使用词向量来填充trainMatrix列表
    #trainMatrix.append(setOfWords2Vec(myVocabList,post))
    trainMatrix.append(bagOfWords2VecMN(myVocabList,post))
p0V,p1V,pAb = trainNB(trainMatrix,listClasses)
print p0V
print '------------------------'
print p1V
print '------------------------'
print pAb#文档属于侮辱类的概率为0.5,该值是正确的
'''

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass):
    p1 = sum(vec2Classify * p1Vec) + log(pClass)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass)
    if p1 > p0: 
        return 1
    else:   
        return 0

def testingNB():
    listPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMatrix=[]          #构建了一个包含所有词的列表
    for post in listPosts:  #使用词向量来填充化& 恤 ^ 列表
        trainMatrix.append(setOfWords2Vec(myVocabList,post))
    p0V,p1V,pAb = trainNB(array(trainMatrix),array(listClasses))

    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))        
    print testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb)

    testEntry = ['stupid','garbage']
    thisDoc  = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb)

#testingNB()

  