# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 17:15:17 2021

@author: Administrator
"""


import os
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

try:
    os.mkdir("figures")
except FileExistsError:
    pass

x = load_boston()
df = pd.DataFrame(x.data, columns = x.feature_names)
print(df.columns.tolist())
df["MEDV"] = x.target
X = df.drop("MEDV",1)   #将模型当中要用到的特征变量保留下来
y = df["MEDV"]          #最后要预测的对象
df.head()

df.dtypes


plt.figure(figsize=(10,8))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.savefig("./figures/heatmap.png", dpi=500)
plt.show()


# 相关系数的值一般是在-1到1这个区间内波动的
# 相关系数要是接近于0意味着变量之间的相关性并不强
# 接近于-1意味着变量之间呈负相关的关系
# 接近于1意味着变量之间呈正相关的关系


# 筛选出与因变量之间的相关性
cor_target = abs(cor["MEDV"])

# 挑选corr大于0.5的相关性系数
relevant_features = cor_target[cor_target>0.5]
relevant_features


print(df[["LSTAT","PTRATIO"]].corr())
print("=" * 50)
print(df[["RM","LSTAT"]].corr())
print("=" * 50)
print(df[["PTRATIO","RM"]].corr())


# 递归消除法
# 我们可以尝试这么一种策略，我们选择一个基准模型，
# 起初将所有的特征变量传进去，我们再确认模型性能的同时通过对特征变量的重要性进行排序，
# 去掉不重要的特征变量，然后不断地重复上面的过程直到达到所需数量的要选择的特征变量。


estimator= LinearRegression()
# 挑选出7个相关的变量
rfe_model = RFE(estimator, n_features_to_select=7, step=1)
# 交给模型去进行拟合
X_rfe = rfe_model.fit_transform(X,y)  
estimator.fit(X_rfe,y)
# 输出各个变量是否是相关的，并且对其进行排序
print(rfe_model.support_)
print(rfe_model.ranking_)


#将13个特征变量都依次遍历一遍
feature_num_list=np.arange(1,13)
# 定义一个准确率
high_score=0
# 最优需要多少个特征变量
num_of_features=0           
score_list =[]
for n in range(len(feature_num_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe_model = RFE(model,n_features_to_select=feature_num_list[n])
    X_train_rfe_model = rfe_model.fit_transform(X_train,y_train)
    X_test_rfe_model = rfe_model.transform(X_test)
    model.fit(X_train_rfe_model,y_train)
    score = model.score(X_test_rfe_model,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        num_of_features = feature_num_list[n]
print("最优的变量是: %d个" %num_of_features)
print("%d个变量的准确率为: %f" % (num_of_features, high_score))


cols = list(X.columns)
model = LinearRegression()
# 初始化RFE模型，筛选出10个变量
rfe_model = RFE(model, n_features_to_select=10, step=1)             
X_rfe = rfe_model.fit_transform(X,y)  
# 拟合训练模型
model.fit(X_rfe,y)              
df = pd.Series(rfe_model.support_,index = cols)
selected_features = df[df==True].index
print(selected_features)


# 正则化
# 例如对于Lasso的正则化而言，对于不相关的特征而言，
# 该算法会让其相关系数变为0，因此不相关的特征变量很快就会被排除掉了，
# 只剩下相关的特征变量
# 可以看到当中有3个特征，‘NOX’、'CHAS'、'INDUS'的相关性为0


lasso = LassoCV()
lasso.fit(X, y)
coef = pd.Series(lasso.coef_, index = X.columns)

print("Lasso算法挑选了 " 
      + str(sum(coef != 0)) 
      + " 个变量，去除掉了" 
      +  str(sum(coef == 0)) + "个变量")


imp_coef = coef.sort_values()
keys = imp_coef.index.tolist()
values = imp_coef.values.tolist()


# 横向条形图，从小到大排序
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

# RGB颜色转16进制颜色
def rgb2hex(R,G,B):
    hexs = [str(x) for x in range(10)] + [chr(x) for x in range(ord('A'), ord('A') + 6)]
    keys = [x for x in range(16)]
    dic = dict(zip(keys, hexs))
    list_R = list(divmod(R, 16))
    hex1 = dic[list_R[0]]+dic[list_R[1]]
    list_G = list(divmod(G, 16))
    hex2 = dic[list_G[0]]+dic[list_G[1]]
    list_B = list(divmod(B, 16))
    hex3 = dic[list_B[0]]+dic[list_B[1]]
    return '#'+hex1+hex2+hex3
    print('#'+hex1+hex2+hex3)

def random_color():
    color_hex = rgb2hex(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    return color_hex

keys.reverse()
values.reverse()
fig, ax = plt.subplots(figsize=(12, 6))
value = ax.barh(keys, values)
plt.barh(range(len(values)), values, color=[random_color() for _ in range(len(keys))], tick_label=keys)
# for key, value in zip(keys, values):
#     ax.text(value + 0.5, key, value, size=12, ha='center', va='bottom')
plt.title("Lasso Model Feature Importance")
plt.xlabel('importance')
plt.ylabel('variables')
plt.savefig("./figures/barh.jpg",dpi=500)
plt.show()



# 下面我们来看一下如何针对离散型的特征变量来做处理，首先我们可以根据缺失值的比重来进行判断
# 要是对于一个离散型的特征变量而言，绝大部分的值都是缺失的，那这个特征变量也就没有存在的必要了
# 我们可以针对这个思路在进行判断。
df = pd.read_excel("./excel/Molecule.xlsx")
df.head()
df.dtypes
func = lambda x: -math.log10(x*10**(-9))
df["pIC50"] = df["IC50 (nM)"].apply(func)
y = df['pIC50']
X = df.drop(columns = ["Index", "Name", "IC50 (nM)", "pIC50"])
feature_names = X.columns.tolist()
missing_series = X.isnull().sum() / X.shape[0]
df = pd.DataFrame(missing_series).rename(columns = {'index': '特征变量', 0: '缺失值比重'})
df.sort_values("缺失值比重", ascending = False).head()
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.figure(figsize = (7, 5))
plt.hist(df['缺失值比重'], bins = np.linspace(0, 1, 11), edgecolor = 'k', color = 'blue', linewidth = 2)
plt.xticks(np.linspace(0, 1, 11))
plt.xlabel('缺失值的比重', size = 14)
plt.ylabel('特征变量的数量', size = 14)
plt.title("缺失值分布图", size = 14)
plt.savefig("./figures/missing.png", dpi=500)
plt.show()



# Select_K_Best算法
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
k = 20
select_regression = SelectKBest(score_func=f_regression, k=k)
z = select_regression.fit_transform(X, y)
filter_2 = select_regression.get_support()
features_regression = np.array(feature_names)
print("所有的特征变量有:")
print(features_regression)
print("筛选出来的{0}个特征变量是:".format(k))
print(features_regression[filter_2])



# Select_K_Best 算法

# 分类问题
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
iris_data = load_iris()
X = iris_data.data
y = iris_data.target
print("数据集的行与列的数量:", X.shape) 
select = SelectKBest(score_func=chi2, k=3)
# 拟合数据
z = select.fit_transform(X,y)
filter_1 = select.get_support()
features = np.array(iris_data.feature_names)
print("所有的特征: ", features)
print("筛选出来最优的特征是: ", features[filter_1])


# 回归问题
boston_data = load_boston()
X = boston_data.data
y = boston_data.target
select_regression = SelectKBest(score_func=f_regression, k=7)
z = select_regression.fit_transform(X, y)
filter_2 = select_regression.get_support()
features_regression = np.array(boston_data.feature_names)
print("所有的特征变量有:")
print(features_regression)
print("筛选出来的7个特征变量则是:")
print(features_regression[filter_2])































