# 导入包
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
import pickle
import numpy as np

# 读取数据
file = pd.read_csv('E:/tutorial/molnet_bace.csv')
dataset = file[['mol', 'AlogP']]
train_data_x, test_data_x, train_y, test_y = train_test_split(dataset['mol'], dataset['AlogP'], test_size = 0.2)

# 计算分子指纹
train_mols = [Chem.MolFromSmiles(smi) for smi in train_data_x] # RDKit Mol object
train_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4, 2048) for mol in train_mols]
train_x = np.asarray(train_fps, dtype = float)

test_mols = [Chem.MolFromSmiles(smi) for smi in test_data_x] # RDKit Mol object
test_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4, 2048) for mol in test_mols]
test_x = np.asarray(test_fps, dtype = float)

# 切分数据

# 格点搜索
# 1.参数字典
svm_param_dict = {'C':[1, 2, 3, 4, 5],
                  'kernel':['poly', 'rbf', 'sigmoid'],
                  'epsilon':[0.1, 0.5, 1.0]}

# 2.分数字典
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

score_dict = {'mse':make_scorer(mean_squared_error),
              'mae':make_scorer(mean_absolute_error),
              'mape':make_scorer(mean_absolute_percentage_error),
              'r2':make_scorer(r2_score)}

# 3.定义模型
svm_reg = svm.SVR()
svm_gs = GridSearchCV(estimator = svm_reg,
                      param_grid = svm_param_dict,
                      scoring = score_dict,
                      n_jobs = 10,
                      cv = 10,
                      refit = 'r2',
                      return_train_score = True)

svm_gs_fit = svm_gs.fit(train_x, train_y)
print("SEARCHING FINISHED!")

best_model = svm_gs_fit.best_estimator_
gs_cv_result = svm_gs_fit.cv_results_
gs_cv_result_df = pd.DataFrame(gs_cv_result)
gs_cv_result_df.to_csv('E:/tutorial/svm_reg_gs_cv_results.csv')

# 交叉验证
svm_best_cv = cross_validate(estimator = best_model,
                             X = train_x,
                             y = train_y,
                             scoring = score_dict,
                             cv = 5,
                             n_jobs = 10,
                             return_train_score = True)

svm_best_cv_df = pd.DataFrame(svm_best_cv)
svm_best_cv_df.to_csv('E:/tutorial/svm_reg_best_cv_results.csv')

svm_cv_train_mae = np.mean(svm_best_cv['train_mae'])
svm_cv_train_mse = np.mean(svm_best_cv['train_mse'])
svm_cv_train_mape = np.mean(svm_best_cv['train_mape'])
svm_cv_train_r2 = np.mean(svm_best_cv['train_r2'])

svm_cv_test_mae = np.mean(svm_best_cv['test_mae'])
svm_cv_test_mse = np.mean(svm_best_cv['test_mse'])
svm_cv_test_mape = np.mean(svm_best_cv['test_mape'])
svm_cv_test_r2 = np.mean(svm_best_cv['test_r2'])

# 用训练好的模型预测外部验证集的结果
svm_ext_pred = best_model.predict(test_x)
svm_ext_pred_df = pd.DataFrame(svm_ext_pred)
test_data_x_list = test_data_x.tolist()
svm_ext_pred_df['mol'] = test_data_x.tolist()
svm_ext_pred_df['true_AlogP'] = test_y.tolist()
svm_ext_pred_df.columns = ['pred_AlogP', 'mol', 'true_AlogP']
svm_ext_pred_df.to_csv('E:/tutorial/svm_reg_ext_results.csv')

# 模型外部验证的性能(performance)
# 1.阳性概率列表
# svm_ext_pred_list = []
# for i, ext_prob in enumerate(svm_ext_pred_prob):
    # ext_pos_prob = ext_prob[1]
    # svm_ext_pred_list.append(ext_pos_prob)

# 2.计算指标
svm_ext_mae = mean_absolute_error(test_y, svm_ext_pred)
svm_ext_mse = mean_squared_error(test_y, svm_ext_pred)
svm_ext_mape = mean_absolute_percentage_error(test_y, svm_ext_pred)
svm_ext_r2 = r2_score(test_y, svm_ext_pred)

svm_perf = {'mae':[svm_cv_test_mae, svm_ext_max_error],
            'mse':[svm_cv_test_mse, svm_ext_mse],
            'mape':[svm_cv_test_mape, svm_ext_mape],
            'r2':[svm_cv_test_r2, svm_ext_r2]}

svm_perf_df = pd.DataFrame.from_dict(svm_perf)
svm_perf_df.index = ['cv', 'ext']
svm_perf_df.to_csv('E:/tutorial/svm_reg_perf.csv', index = True)

# 存储已经训练好的模型
with open('E:/tutorial/svm_reg.pkl', 'wb') as file:
    pickle.dump(best_model, file)
