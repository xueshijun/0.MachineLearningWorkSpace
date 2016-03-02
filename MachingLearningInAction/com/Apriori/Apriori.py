# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Mon Feb 22 09:22:52 2016
#Version: 1.0
#-------------------------------------------------------------------------------
'''
关联分析是用于发现大数据集中元素间有趣关系的一个工具集,
可以采用两种方式来量化这些有趣的关系:
    1.使用频繁项集，它会给出经常在一起出现的元素项。
    2.关联规则，每条关联规则意味着元素项之间的“ 如果……那么”关系。


Apriori算法,
优点：易编码实现。'
缺点：在大数据集上可能较慢。
适用数据类型：数值型或者标称型数据。

Apriori算法的一般过程
    (1)收集数据：使用任意方法。
    (2)准备数据：任何数据类型都可以，因为我们只保存集合。
    (3)分析数据：使用任意方法。
    (4)训练算法：使用Apriori算法来找到频繁项集。
    (5)测试算法：不需要测试过程。
    (6)使用算法：用于发现频繁项集以及物品之间的关联规则。
'''

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

#将构建集合C1,即大小为1的所有候选项集的集合。
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                '''添加只包含该物品项的一个列表,目的是为每个物品项构建一个集合。
                即C1是一个集合的集合，如 {{0},{1},{2},...}，每次添加的都是单个项构成的集合{0}、{1}、{2}...。
                在Apriori算法的后续处理中，需要做集合操作。
                Python不能创建只有一个整数的集合，因此这里实现必须使用列表。
                这就是使用一个由单物品列表组成的大列表的原因。
                '''
                C1.append([item])
    C1.sort()
    '''是指被"冰冻"的集合，就是说它们是不可改变的，即用户不能修改它们。
    这里必须要使用frozenset而不是set类型，因为之后必须要将这些集合作为字典键值使用,
    使用frozenset可以实现这一点，而set却做不到。'''
    return map(frozenset, C1)#use frozen set so we can use it as a key in a dict    

dataSet = loadDataSet()
C1 = createC1(dataSet)
print C1    #[frozenset([1]), frozenset([2]), frozenset([3]), frozenset([4]), frozenset([5])]



'''数据集扫描的伪代码:
对数据集中的每条交易记录tran
对每个候选项集can：
    检查一下can是否是tran的子集：
    如果是，则增加can的计数值
对每个候选项集：
如果其支持度不低于最小值，则保留该项集
返回所有频繁项集列表'''
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for transaction in D:
        for can in Ck:
            if can.issubset(transaction):
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1
    retList = [];    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/ float(len(D))
        if support >= minSupport: retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

D = map(set,dataSet)
print D    #[set([1, 3, 4]), set([2, 3, 5]), set([1, 2, 3, 5]), set([2, 5])]

#有了集合形式的数据,去掉那些不满足最小支持度的项集
L1,suppData = scanD(D,C1,0.5)
print L1#[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]

'''整个Apriori算法的伪代码如下：
当集合中项的个数大于0时
    构建一个k个项组成的候选项集的列表
    检查数据以确认每个项集都是频繁的
    保留频繁项集并构建奸1项组成的候选项集的列表
'''


'''
创建候选项集Ck
函数以{0}、{1}、{2}作为输入，会生成{0,1}、{0,2}以及{1,2}。

关于k-2:
如果利用{0,1},{0,2},{1,2}来创建三元素项集，
    如果将每两个集合合并，就会得到{0,1,2},{0,1,2},{0,l,2},
        同样的结果集合会重复3次。
    如果比较集合{0,1},{0,2}, {1,2}的第1个元素并只对第1个元素相同的集合求并操作，
        得到{0, l,2},而且只有一次操作！这样就不需要遍历列表来寻找非重复值。
    即比较前K-1个,索引为k-2
'''
def aprioriGen(Lk, k): #creates Ck
    retList = [] 
    for i in range(len(Lk)):
        for j in range(i+1, len(Lk)):  
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #两个集合的前面k-2个元素都相等，那么就将这两个集合合成一个大小为k的集合
                retList.append(Lk[i] | Lk[j]) #set union
    return retList
'''
Apriori原理是说:
    如果某个项集是频繁的，那么它的所有子集也是频繁的。
    如果一个项集是非频繁集，那么它的所有超集也是非频繁的
'''
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

L, supportData = apriori(dataSet)
print '----------------'

print aprioriGen(L[0],2)


#支持度0.7
'''如果某条规则并不满足最小可信度要求，那么该规则的所有子集也不会满足最小可信度要求'''
L, supportData = apriori(dataSet,minSupport=0.7)
#print supportData
#for i in range(len(L)):
#    print L[i],supportData[i]

'''
首先从一个频繁项集开始，接着创建一个规则列表，其中规则右部只包含一个元素，然后对这些规则进行测试。
接下来合并所有剩余规则来创建一个新的规则列表，其中规则右部包含两个元素。这种方法也被称作分级法


频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
'''
#L:频繁项集列表,supportData:包含那些频繁项集支持数据的字典,minConf:最小可信度阈值
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):#只获取有两个或更多元素的集合
        for freqSet in L[i]:            
            #该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表Hl
            #因为无法从单元素项集中构建关联规则，所以要从包含两个或者更多元素的项集开始规则构建过程
            #{0,1,2}开始，那么H1应该是[{0},{1},{2}]
            H1 = [frozenset([item]) for item in freqSet] 
            if (i > 1): 
                #如果频繁项集的元素数目超过2,那么会考虑对它做进一步的合并。
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                #如果项集中只有两个元素，那么使用函数calcConf()来计算可信度值。
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         
#对规则进行评估
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH
#生成候选规则集合
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
print '========================'
L, supportData = apriori(dataSet,minSupport=0.5)
rules = generateRules(L, supportData,minConf=0.7)
print rules