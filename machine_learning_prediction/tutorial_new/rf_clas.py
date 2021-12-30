import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import pickle

# 读取文件
train_set = pd.read_csv('E:/tutorial/toy_set.csv')
ext_set = pd.read_csv('E:/tutorial/toy_ext_set.csv')

# 计算训练集分子指纹
train_mols = [Chem.MolFromSmiles(smi) for smi in train_set['smiles']]
train_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4, 2048) for mol in train_mols]
train_fps_array = np.asarray(train_fps, dtype = float)
train_class = train_set['mutagenicity']

# 格点搜索
# 1.参数字典
rf_param_dict = {'n_estimators':[50, 100, 150, 200, 500],
                 'max_depth':[10, 50, 100],
                 'max_features': ["auto","sqrt","log2"]}

# 2.性能指标字典
score_dict = {'AUC':'roc_auc',
              'ACC': make_scorer(accuracy_score),
              'SEN': make_scorer(recall_score)}

# 3.格点搜索定义
rf_classifier = RandomForestClassifier()
rf_gs = GridSearchCV(estimator = rf_classifier,
                     param_grid = rf_param_dict,
                     scoring = score_dict,
                     n_jobs = 10,
                     cv = 5, 
                     refit = 'AUC',
                     return_train_score = True) # 可以查看模型是否过拟合

rf_gs_fit = rf_gs.fit(train_fps_array, train_class)
print("********************FINISHED********************")

rf_best_model = rf_gs_fit.best_estimator_
rf_gs_cv_result = rf_gs_fit.cv_results_
rf_gs_cv_result_df = pd.DataFrame(rf_gs_cv_result)
rf_gs_cv_result_df.to_csv('E:/tutorial/rf_gs_cv_results.csv')

# 交叉验证
rf_best_cv = cross_validate(estimator = rf_best_model,
                            X = train_fps_array,
                            y = train_class,
                            scoring = score_dict,
                            cv = 5,
                            n_jobs = 10,
                            return_train_score = True)

rf_best_cv_df = pd.DataFrame(rf_best_cv)
rf_best_cv_df.to_csv('E:/tutorial/rf_gs_cv_results.csv')

rf_cv_train_auc = np.mean(rf_best_cv['train_AUC'])
rf_cv_train_acc = np.mean(rf_best_cv['train_ACC'])
rf_cv_train_sen = np.mean(rf_best_cv['train_SEN'])
rf_cv_train_spc = (rf_cv_train_acc * len(train_class) - rf_cv_train_sen * train_class.sum())/(len(train_class)-train_class.sum())

rf_cv_test_auc = np.mean(rf_best_cv['test_AUC'])
rf_cv_test_acc = np.mean(rf_best_cv['test_ACC'])
rf_cv_test_sen = np.mean(rf_best_cv['test_SEN'])
rf_cv_test_spc = (rf_cv_test_acc * len(train_class) - rf_cv_test_sen * train_class.sum())/(len(train_class)-train_class.sum())

# 计算外部验证集的分子指纹
ext_mols = [Chem.MolFromSmiles(smi) for smi in ext_set['smiles']]
ext_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4, 2048) for mol in ext_mols]
ext_fps_array = np.asarray(ext_fps, dtype = float)
ext_class = ext_set['mutagenicity']

# 外部验证
rf_ext_pred_prob = rf_best_model.predict_proba(ext_fps_array)
rf_ext_pred_prob_df = pd.DataFrame(rf_ext_pred_prob)
rf_ext_pred_prob_df['smiles'] = ext_set['smiles']
rf_ext_pred_prob_df.columns = ['neg', 'pos', 'smiles']
rf_ext_pred_prob_df.to_csv('E:/tutorial/rf_ext_prob_results.csv')

# 模型泛化性能
# 1.阳性概率列表
rf_ext_pred_list = []
for rf_pos in rf_ext_pred_prob:
    rf_pos_prob = rf_pos[1]
    rf_ext_pred_list.append(rf_pos_prob)

# 2.计算指标
# rf_ext_pred_array = np.array(rf_ext_pred_list)
rf_ext_auc = roc_auc_score(ext_class, rf_ext_pred_list)
rf_ext_acc = accuracy_score(ext_class, np.round(rf_ext_pred_list))
rf_ext_sen = recall_score(ext_class, np.round(rf_ext_pred_list))
rf_ext_spc = (rf_ext_acc * len(ext_class) - rf_ext_sen * ext_class.sum())/(len(ext_class)-ext_class.sum())

rf_perf = {'AUC':[rf_cv_test_auc, rf_ext_auc],
           'ACC':[rf_cv_test_acc, rf_ext_acc],
           'SEN':[rf_cv_test_sen, rf_ext_sen],
           'SPC':[rf_cv_test_spc, rf_ext_sen]}

rf_perf_df = pd.DataFrame.from_dict(rf_perf)
rf_perf_df.to_csv('E:/tutorial/rf_performance.csv')

# 存储已经训练好的模型
with open('E:/tutorial/rf.pkl', 'wb') as file:
    pickle.dump(rf_best_model, file)
