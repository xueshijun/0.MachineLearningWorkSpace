# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#Name    :
#Author : Xueshijun
#MailTo : xueshijun_2010@163.com    / 324858038@qq.com
#QQ     : 324858038
#Blog   : http://blog.csdn.net/xueshijun666
#Created on Thu Apr 14 20:25:09 2016
#Version: 1.0
#-------------------------------------------------------------------------------
import networkx as net
import matplotlib.pyplot as plot
orgchart = net.read_pajek("data/ACME_orgchart.net")
net.draw(orgchart)