# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 22:41:47 2021

@author: weineng.zhou
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

try:
    os.mkdir('figures')
except FileExistsError:
    pass


# RGB颜色转16进制颜色
def rgb2hex(R, G, B):
    hexs = [str(x) for x in range(10)] + [chr(x) for x in range(ord('A'), ord('A') + 6)]
    keys = [x for x in range(16)]
    dic = dict(zip(keys, hexs))
    list_R = list(divmod(R, 16))
    hex1 = dic[list_R[0]] + dic[list_R[1]]
    list_G = list(divmod(G, 16))
    hex2 = dic[list_G[0]] + dic[list_G[1]]
    list_B = list(divmod(B, 16))
    hex3 = dic[list_B[0]] + dic[list_B[1]]
    return '#' + hex1 + hex2 + hex3

def random_color():
    color_hex = rgb2hex(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color_hex


# import some data to play with
iris = datasets.load_iris()
filename = './excel/cbdd13494-sup-0002-tables1.xlsx'
df = pd.read_excel(filename, sheet_name='data', header=0, 
                   skiprows=list(range(1)), 
                   usecols=list(range(32)), 
                   keep_default_na=False,
                   na_values=['NA'])
arr = np.array(df.iloc[:,1:])
X = arr[:,:-1]
y = arr[:,-1]


# 前两个特征值
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
plt.figure(2, figsize=(8, 6))
plt.clf()
target_names = ["active", "inactive"]
colors = ["red", "blue"]
lw = 0.8
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1],
                color=color, 
                alpha=.8, lw=lw, 
                label=target_name)
plt.xlabel("chi0v")
plt.ylabel("SlogP_VSA9")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./figures/PCA.jpg', dpi=500)
plt.show()


# 主成分分析
fig = plt.figure(figsize=(8, 6))
target_names = np.array(["inactive", "active"])
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
plt.figure()
colors = ["red", "blue"]
lw = 0.8
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], 
                color=color,
                alpha=.8,
                lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of dataset')
plt.xlabel('X1', fontsize=10)
plt.ylabel('X2', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('./figures/PCA2.jpg', dpi=500)
plt.show()


# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X)

colors = ["red", "blue"]
lw = 0.8
for color, i, target_name in zip(colors, [0, 1], target_names):
    ax.scatter(
        X_reduced[y == i, 0],
        X_reduced[y == i, 1],
        X_reduced[y == i, 2],
        color=color,
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
        alpha=.8,
        lw=lw, 
        label=target_name
)

ax.set_title("Three PCA directions")
ax.set_xlabel("PCA1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("PCA2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("PCA3")
ax.w_zaxis.set_ticklabels([])
ax.grid(False)
plt.savefig('./figures/PCA3D.jpg', dpi=500)
plt.show()







