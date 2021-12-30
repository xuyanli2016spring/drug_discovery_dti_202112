"""
# -*- coding: utf-8 -*-
Created on Mon May  4 14:25:12 2021
@author: Alan
"""

import itertools
from itertools import cycle
from scipy import interp
import warnings
warnings.filterwarnings('ignore')

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize

#模型选择
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
#支持sklearn>=0.24
#from sklearn.model_selection import HalvingGridSearchCV
#from sklearn.model_selection import HalvingRandomSearchCV

from sklearn.experimental import enable_halving_search_cv  

# 机器学习算法
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.multiclass import OneVsRestClassifier

# 集成学习机器学习方法
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import AdaBoostClassifier

# 极端梯度提升树与轻量梯度提升机
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier


#模型评估
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, plot_roc_curve
from sklearn.metrics import roc_auc_score

#特征的重要性
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.inspection import permutation_importance


import os
try:
    os.mkdir("output")
except FileExistsError:
    pass

# global variable
date       = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
author     = 'Alan'
ML_method  = 'Random Forest'
abbreviate = 'RF'
dev_data   = './excel/dev.xlsx'
val_data   = './excel/val.xlsx'

class_names = ['非有效药物', '有效药物']
t1 = datetime.datetime.now()
print('开始时间:', t1.strftime('%Y-%m-%d %H:%M:%S'))


###############################################################################
# load the data
a = pd.read_excel('{}'.format(dev_data))
feature_names = a.columns.tolist()[:-1]
feature_names = np.array(feature_names)

a = np.array(a)

#get independent variable
X = a[:,:-1]

#get the dependent variable
y = a[:,-1]

# Binarize the output
y = label_binarize(y, classes=[0., 1.])

#n_classes = y.shape[1]
n_classes = 2


# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)

# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                 random_state=random_state)


base_estimator = RandomForestClassifier(random_state=random_state)

# max_depth = [3, 5, 10]
# min_samples_split = [2, 5, 10]
n_estimators = [200, 300, 500]
param_grid = {'n_estimators': n_estimators}


# param_grid = {'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'n_estimators': n_estimators}

# sh = HalvingGridSearchCV(base_estimator, 
#                          param_grid, cv=5, 
#                          factor=2, 
#                          resource='n_estimators',
#                          max_resources=30).fit(X_train, y_train.ravel())
# print(sh.best_estimator_)


# 超参数优化
scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for {}".format(score))
    print()
    classifier = GridSearchCV(base_estimator, 
                              param_grid, cv=5, 
                              scoring='{0}'.format(score)
                              )
    
    classifier.fit(X_train, y_train.ravel())
    print("Best parameters set found on development set:")
    print()
    print(classifier.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, classifier.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


# Learn to predict each class against the other
classifier = RandomForestClassifier(n_estimators=500, random_state=0)


# fit the model
classifier.fit(X_train, y_train.ravel())
print("Accuracy on test data: {:.2f}".format(classifier.score(X_test, y_test)))


# 特征的重要性
result = permutation_importance(classifier, X_train, y_train, 
                                n_repeats=10,
                                random_state=0)

perm_sorted_idx = result.importances_mean.argsort()

tree_importance_sorted_idx = np.argsort(classifier.feature_importances_)
tree_indices = np.arange(0, len(classifier.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.barh(tree_indices,
         classifier.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticks(tree_indices)
ax1.set_yticklabels(feature_names[tree_importance_sorted_idx])
ax1.set_ylim((0, len(classifier.feature_importances_)))
ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
            labels=feature_names[perm_sorted_idx])
fig.tight_layout()
plt.show()


#特征的共线性问题：分层聚类，选择一个阈值，并从每个聚类中保留一个特征
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

corr = spearmanr(X_train).correlation

corr_linkage = hierarchy.ward(corr)

dendro = hierarchy.dendrogram(
    corr_linkage, labels=feature_names.tolist(), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro['ivl']))

ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
fig.tight_layout()
plt.show()


# predic by X_train
y_train_pred = classifier.predict(X_train)
# predict by X_test
y_test_pred = classifier.predict(X_test)


# 模型评估
#the label of confusion matrix
# class_names = np.array(class_names)
# plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center", fontsize=20,
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()

# calculate the training set confusion matrix
cnf_matrix = confusion_matrix(y_test, y_test_pred)
np.set_printoptions(precision=2)
# without normalization confusion matrix
plt.figure(figsize=(20, 10))
plot_confusion_matrix(cnf_matrix, classes=class_names,title='test eet')
plt.savefig('./output/test_matrix_{}.tiff'.format(abbreviate), dpi=500)
plt.show()


# cutoff: 预测概率
y_test_score = classifier.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()


# label 0
fpr[0], tpr[0], _ = roc_curve(y_test[:, 0], y_test_score[:, 0])
roc_auc[0] = auc(fpr[0], tpr[0])

# label 1
fpr[1], tpr[1], _ = roc_curve(y_test[:, 0], y_test_score[:, 1])
roc_auc[1] = auc(fpr[1], tpr[1])


# 对于三分类以上
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_train[:, i], y_train_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])


# Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(fpr[1], tpr[1], color='red', lw=lw, label='ROC curve (area = {:.2f})'.format(roc_auc[1]))
plt.plot([0, 1], [0, 1], color='#00bc57', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for test set')
plt.legend(loc="lower right")
plt.savefig("./output/testROC.tiff", dpi=500)
plt.show()


########################################################################################################################
# 10-fold cross validation for accuracy
from sklearn.model_selection import cross_val_score
import seaborn as sns
scores = cross_val_score(classifier, X, y, cv=10)
print(scores)
print(scores.mean())
scores_df = pd.DataFrame(scores)
name = [''.format(abbreviate)]*10
name_df = pd.DataFrame(name)
M = pd.concat([name_df, scores_df], axis=1) #横向拼接数据框
M.columns=['Model', 'Accuracy']
M.to_excel('./output/{}_Accuracy.xlsx'.format(abbreviate), index=False)
sns.boxplot(data=M, x = 'Model', y = 'Accuracy', color='#00b8e5')
plt.savefig("./output/boxplot.jpg", dpi=600)
########################################################################################################################


########################################################################################################################
# 10-fold cross validation for ROC

#load the data
a = pd.read_excel('{}'.format(dev_data))
a = np.array(a)
#get independent variable
X = a[:,:-1]
#get the dependent variable
y = a[:,-1]
n_samples, n_features = X.shape
# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
# Classification and ROC analysis
cv = StratifiedKFold(n_splits=10)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = plot_roc_curve(classifier, X[test], y[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05],
       ylim=[-0.05, 1.05],
       xlabel='False Positive Rate (1-Specificity)',
       ylabel='Ture Positive Rate (Sensitivity)',
       title='{}'.format(ML_method))
ax.legend(loc='lower right')
plt.savefig('./output/roc_crossval_{}.tiff'.format(abbreviate), dpi=500)
plt.show()




