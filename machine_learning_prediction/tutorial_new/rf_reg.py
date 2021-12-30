import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import pickle

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

# 格点搜索
# 1.参数字典
rf_param_dict = {'n_estimators':[50, 100, 150, 200],
                 'max_depth':[10, 50, 100],
                 'max_features': ["auto","sqrt","log2"]}

# 2.性能指标字典
score_dict = {'mse':make_scorer(mean_squared_error),
              'mae':make_scorer(mean_absolute_error),
              'mape':make_scorer(mean_absolute_percentage_error),
              'r2':make_scorer(r2_score)}


# 3.格点搜索定义
rf_reg = RandomForestRegressor()
rf_gs = GridSearchCV(estimator = rf_reg,
                     param_grid = rf_param_dict,
                     scoring = score_dict,
                     n_jobs = 10,
                     cv = 5, 
                     refit = 'r2',
                     return_train_score = True) # 可以查看模型是否过拟合

rf_gs_fit = rf_gs.fit(train_x, train_y)
print("********************FINISHED********************")

rf_best_model = rf_gs_fit.best_estimator_
rf_gs_cv_result = rf_gs_fit.cv_results_
rf_gs_cv_result_df = pd.DataFrame(rf_gs_cv_result)
rf_gs_cv_result_df.to_csv('E:/tutorial/rf_gs_cv_results.csv')

# 交叉验证
rf_best_cv = cross_validate(estimator = rf_best_model,
                            X = train_x,
                            y = train_y,
                            scoring = score_dict,
                            cv = 5,
                            n_jobs = 10,
                            return_train_score = True)

rf_best_cv_df = pd.DataFrame(rf_best_cv)
rf_best_cv_df.to_csv('E:/tutorial/rf_reg_gs_cv_results.csv')

rf_cv_train_mae = np.mean(rf_best_cv['train_mae'])
rf_cv_train_mse = np.mean(rf_best_cv['train_mse'])
rf_cv_train_mape = np.mean(rf_best_cv['train_mape'])
rf_cv_train_r2 = np.mean(rf_best_cv['train_r2'])

rf_cv_test_mae = np.mean(rf_best_cv['test_mae'])
rf_cv_test_mse = np.mean(rf_best_cv['test_mse'])
rf_cv_test_mape = np.mean(rf_best_cv['test_mape'])
rf_cv_test_r2 = np.mean(rf_best_cv['test_r2'])

# 用训练好的模型预测外部验证集的结果
rf_ext_pred = rf_best_model.predict(test_x)
rf_ext_pred_df = pd.DataFrame(rf_ext_pred)
test_data_x_list = test_data_x.tolist()
rf_ext_pred_df['mol'] = test_data_x.tolist()
rf_ext_pred_df['true_AlogP'] = test_y.tolist()
rf_ext_pred_df.columns = ['pred_AlogP', 'mol', 'true_AlogP']
rf_ext_pred_df.to_csv('E:/tutorial/rf_reg_ext_results.csv')

# 计算指标
rf_ext_mae = mean_absolute_error(test_y, rf_ext_pred)
rf_ext_mse = mean_squared_error(test_y, rf_ext_pred)
rf_ext_mape = mean_absolute_percentage_error(test_y, rf_ext_pred)
rf_ext_r2 = r2_score(test_y, rf_ext_pred)

rf_perf = {'mae':[rf_cv_test_mae, rf_ext_mae],
            'mse':[rf_cv_test_mse, rf_ext_mse],
            'mape':[rf_cv_test_mape, rf_ext_mape],
            'r2':[rf_cv_test_r2, rf_ext_r2]}

rf_perf_df = pd.DataFrame.from_dict(rf_perf)
rf_perf_df.index = ['cv', 'ext']
rf_perf_df.to_csv('E:/tutorial/rf_reg_perf.csv', index = True)

# 存储已经训练好的模型
with open('E:/tutorial/rf_reg.pkl', 'wb') as file:
    pickle.dump(rf_best_model, file)
