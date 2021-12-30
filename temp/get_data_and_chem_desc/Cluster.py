# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:50:13 2018

@author: zwn
"""


import pandas as pd
import seaborn as sns  #用于画热图的工具包
from scipy.cluster import hierarchy  #用于进行层次聚类，话层次聚类图的工具包
from scipy import cluster   
import matplotlib.pyplot as plt
from sklearn import decomposition as skldec #用于主成分分析降维的包

df = pd.read_excel("./excel/dev.xlsx")
df = df.iloc[:,1:-2]
df.shape

#层次聚类的热图和聚类图
sns.clustermap(df,method ='ward',metric='euclidean')


#层次聚类树状图    
Z = hierarchy.linkage(df, method ='ward',metric='euclidean')
hierarchy.dendrogram(Z, labels = df.index)
#在某个高度进行剪切
label = cluster.hierarchy.cut_tree(Z,height=0.8)
label = label.reshape(label.size)




