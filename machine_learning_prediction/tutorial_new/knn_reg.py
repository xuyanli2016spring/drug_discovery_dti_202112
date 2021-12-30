import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
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

# 格点搜索
# 1.参数字典
knn_param_dict = {'n_neighbors':[5, 10, 15, 20],
                  'weights':['uniform', 'distance']}

# 2.分数字典
score_dict = {'mse':make_scorer(mean_squared_error),
              'mae':make_scorer(mean_absolute_error),
              'mape':make_scorer(mean_absolute_percentage_error),
              'r2':make_scorer(r2_score)}

# 3.定义模型
knn_reg = KNeighborsRegressor()
knn_gs = GridSearchCV(estimator = knn_reg,
                      param_grid = knn_param_dict,
                      scoring = score_dict,
                      n_jobs = 10,
                      cv = 10,
                      refit = 'r2',
                      return_train_score = True)

knn_gs_fit = knn_gs.fit(train_x, train_y)
print("SEARCHING FINISHED!")

best_model = knn_gs_fit.best_estimator_
gs_cv_result = knn_gs_fit.cv_results_
gs_cv_result_df = pd.DataFrame(gs_cv_result)
gs_cv_result_df.to_csv('E:/tutorial/knn_reg_gs_cv_results.csv')

# 交叉验证
knn_best_cv = cross_validate(estimator = best_model,
                             X = train_x,
                             y = train_y,
                             scoring = score_dict,
                             cv = 5,
                             n_jobs = 10,
                             return_train_score = True)

knn_best_cv_df = pd.DataFrame(knn_best_cv)
knn_best_cv_df.to_csv('E:/tutorial/knn_best_cv_results.csv')

knn_cv_train_mae = np.mean(knn_best_cv['train_mae'])
knn_cv_train_mse = np.mean(knn_best_cv['train_mse'])
knn_cv_train_mape = np.mean(knn_best_cv['train_mape'])
knn_cv_train_r2 = np.mean(knn_best_cv['train_r2'])

knn_cv_test_mae = np.mean(knn_best_cv['test_mae'])
knn_cv_test_mse = np.mean(knn_best_cv['test_mse'])
knn_cv_test_mape = np.mean(knn_best_cv['test_mape'])
knn_cv_test_r2 = np.mean(knn_best_cv['test_r2'])

# 外部验证
knn_ext_pred = best_model.predict(test_x)
knn_ext_pred_df = pd.DataFrame(knn_ext_pred)
test_data_x_list = test_data_x.tolist()
knn_ext_pred_df['mol'] = test_data_x.tolist()
knn_ext_pred_df['true_AlogP'] = test_y.tolist()
knn_ext_pred_df.columns = ['pred_AlogP', 'mol', 'true_AlogP']
knn_ext_pred_df.to_csv('E:/tutorial/knn_reg_ext_results.csv')

knn_ext_mae = mean_absolute_error(test_y, knn_ext_pred)
knn_ext_mse = mean_squared_error(test_y, knn_ext_pred)
knn_ext_mape = mean_absolute_percentage_error(test_y, knn_ext_pred)
knn_ext_r2 = r2_score(test_y, knn_ext_pred)

knn_perf = {'mae':[knn_cv_test_mae, knn_ext_mae],
            'mse':[knn_cv_test_mse, knn_ext_mse],
            'mape':[knn_cv_test_mape, knn_ext_mape],
            'r2':[knn_cv_test_r2, knn_ext_r2]}

knn_perf_df = pd.DataFrame.from_dict(knn_perf)
knn_perf_df.index = ['cv', 'ext']
knn_perf_df.to_csv('E:/tutorial/knn_reg_perf.csv', index = True)

with open('E:/tutorial/knn_reg.pkl', 'wb') as file:
    pickle.dump(best_model, file)

