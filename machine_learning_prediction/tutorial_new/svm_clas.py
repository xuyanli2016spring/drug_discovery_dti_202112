# 导入包
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
import pickle
import numpy as np

# 读取数据
train_set = pd.read_csv('E:/tutorial/toy_set.csv')
ext_set = pd.read_csv('E:/tutorial/toy_ext_set.csv')

# 计算训练集的分子指纹
train_mols = [Chem.MolFromSmiles(smi) for smi in train_set['smiles']] # RDKit Mol object
train_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(train_mol, 4, 2048) for train_mol in train_mols]
train_fps_array = np.asarray(train_fps, dtype = float)
train_class = train_set['mutagenicity']

# 格点搜索
# 1.参数字典
svm_param_dict = {'C':[0.1, 0.5, 1, 2, 3, 4, 5],
                  'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                  'probability':[True]}

# 2.分数字典
score_dict = {'AUC':'roc_auc',
              'ACC':make_scorer(accuracy_score),
              'SEN':make_scorer(recall_score)}

# 3.定义模型
svm_classifier = SVC()
svm_gs = GridSearchCV(estimator = svm_classifier, param_grid = svm_param_dict, scoring = score_dict, n_jobs = 10, cv = 10, refit = 'AUC', return_train_score = True)

svm_gs_fit = svm_gs.fit(train_fps_array, train_class)
print("SEARCHING FINISHED!")

best_model = svm_gs_fit.best_estimator_
gs_cv_result = svm_gs_fit.cv_results_
gs_cv_result_df = pd.DataFrame(gs_cv_result)
gs_cv_result_df.to_csv('E:/tutorial/svm_gs_cv_results.csv')

# 交叉验证
svm_best_cv = cross_validate(estimator = best_model,
                             X = train_fps_array,
                             y = train_class,
                             scoring = score_dict,
                             cv = 5,
                             n_jobs = 10,
                             return_train_score = True)

svm_best_cv_df = pd.DataFrame(svm_best_cv)
svm_best_cv_df.to_csv('E:/tutorial/svm_best_cv_results.csv')
svm_cv_train_auc = np.mean(svm_best_cv['train_AUC'])
svm_cv_test_auc = np.mean(svm_best_cv['test_AUC'])
svm_cv_train_acc = np.mean(svm_best_cv['train_ACC'])
svm_cv_test_acc = np.mean(svm_best_cv['test_ACC'])
svm_cv_train_sen = np.mean(svm_best_cv['train_SEN'])
svm_cv_test_sen = np.mean(svm_best_cv['test_SEN'])
svm_cv_train_spc = (svm_cv_train_acc * len(train_class) - svm_cv_train_sen * train_class.sum())/(len(train_class)-train_class.sum())
svm_cv_test_spc = (svm_cv_test_acc * len(train_class) - svm_cv_test_sen * train_class.sum())/(len(train_class)-train_class.sum())

# 外部验证
ext_mols = [Chem.MolFromSmiles(smi) for smi in ext_set['smiles']]
ext_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4, 2048) for mol in ext_mols]
ext_fps_array = np.asarray(ext_fps, dtype = float)
ext_class = ext_set['mutagenicity']

# 用训练好的模型预测外部验证集的结果
svm_ext_pred_prob = best_model.predict_proba(ext_fps_array)
svm_ext_pred_prob_df = pd.DataFrame(svm_ext_pred_prob)
svm_ext_pred_prob_df['smiles'] = ext_set['smiles']
svm_ext_pred_prob_df.columns = ['negative', 'positive', 'smiles']
svm_ext_pred_prob_df.to_csv('E:/tutorial/svm_ext_prob_results.csv')

# 模型外部验证的性能(performance)
# 1.阳性概率列表
svm_ext_pred_list = []
for i, ext_prob in enumerate(svm_ext_pred_prob):
    ext_pos_prob = ext_prob[1]
    svm_ext_pred_list.append(ext_pos_prob)

# 2.计算指标
svm_ext_auc = roc_auc_score(ext_class, svm_ext_pred_list)
svm_ext_acc = accuracy_score(ext_class, np.round(svm_ext_pred_list))
svm_ext_sen = recall_score(ext_class, np.round(svm_ext_pred_list))
svm_ext_spc = (svm_ext_acc * len(ext_class)- svm_ext_sen * ext_class.sum())/(len(ext_class)-ext_class.sum())

svm_perf = {'AUC':[svm_cv_test_auc, svm_ext_auc],
            'ACC':[svm_cv_test_acc, svm_ext_acc],
            'SEN':[svm_cv_test_sen, svm_ext_sen],
            'SPC':[svm_cv_test_spc, svm_ext_spc]}

svm_perf_df = pd.DataFrame.from_dict(svm_perf)

# 存储已经训练好的模型
with open('E:/tutorial/svm.pkl', 'wb') as file:
    pickle.dump(best_model, file)
