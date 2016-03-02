# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Tue Feb 23 10:40:36 2016
#Version: 1.0
#-------------------------------------------------------------------------------
'''
http://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
http://votesmart.org/

示例:在美国国会投票记录中发现关联规则
(1)收集数据：使用votesmart模块来访问投票记录
(2)准备数据：构造一个函数来将投票转化为一串交易记录。
(3)分析数据：在Python提示符下查看准备的数据以确保其正确性。
(4)训练算法：使用本章早先的apriori()和generateRules()函数来发现投票记录中的有趣信息。
(5)测试算法：不适用，即没有测试过程。
(6)使用算法：这里只是出于娱乐的目的，不过也可以使用分析结果来为政治竞选活动服务 ，或者预测选举官员会如何投票。


'''
#==============================================================================
# from votesmart import votesmart
# votesmart.apikey='49024thereoncewasamanfromnantucket94040'
# 
# 
#     
# 
#==============================================================================
  
from time import sleep
from votesmart import votesmart
#==============================================================================
# votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
# #votesmart.apikey = 'get your api key first'
# bills =votesmart.votes.getBillsByStateRecent()
# for bill in bills:
#     print bill.title,bill.billId
#获得每条议案的更多内容
#bill = votesmart.votes.getBill(11820)
#为获得某条特定议案的投票信息
#voteList = votesmart.votes.getBillActionVotes(31670)
#==============================================================================

#收集美国国会议案中actionID的函数     
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:#api call
            billDetail = votesmart.votes.getBill(billNum) 
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print "problem getting bill %d" % billNum
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList
actionIdList, billTitleList = getActionIds()

#基于投票数据的事务列表填充函数
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print 'getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print "problem getting actionId: %d" % actionId
        voteCount += 2
    return transDict, itemMeaning

transDict,itemMeaning = getTransList(actionIdList[:2],billTitleList[:2])
dataSet = [transDict[key] for key in transDict.keys()]

