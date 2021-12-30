import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import sklearn
import xgboost as xgb
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
xgb_param_grid = {'n_estimators':[50, 100],
                  'max_depth':[6, 8, 10]}

# 2.性能指标字典
score_dict = {'mse':make_scorer(mean_squared_error),
              'mae':make_scorer(mean_absolute_error),
              'mape':make_scorer(mean_absolute_percentage_error),
              'r2':make_scorer(r2_score)}

print('Grid search of XGBoost...')
xgb_reg = xgb.XGBRegressor()
xgb_gs = GridSearchCV(xgb_reg,
                      xgb_param_grid,
                      scoring = score_dict,
                      cv = 5,
                      n_jobs = 10,
                      refit = 'r2',
                      return_train_score = True)

xgb_gs_ecfp = xgb_gs.fit(train_x, train_y)
xgb_model = xgb_gs_ecfp.best_estimator_

print('Cross Validation of XGBoost...')
xgb_cv = cross_validate(xgb_model,
                        train_x,
                        train_y,
                        cv = 5,
                        n_jobs = 10,
                        scoring = score_dict,
                        return_train_score = True)

print('Calculate metrics of XGBoost cross validation...')
xgb_cv_train_mae = np.mean(xgb_cv['train_mae'])
xgb_cv_train_mse = np.mean(xgb_cv['train_mse'])
xgb_cv_train_mape = np.mean(xgb_cv['train_mape'])
xgb_cv_train_r2 = np.mean(xgb_cv['train_r2'])

xgb_cv_test_mae = np.mean(xgb_cv['test_mae'])
xgb_cv_test_mse = np.mean(xgb_cv['test_mse'])
xgb_cv_test_mape = np.mean(xgb_cv['test_mape'])
xgb_cv_test_r2 = np.mean(xgb_cv['test_r2'])

# 用训练好的模型预测外部验证集的结果
xgb_ext_pred = xgb_model.predict(test_x) # predict_prob()
# xgb_ext_pred_df = pd.DataFrame(xgb_ext_pred)
# test_data_x_list = test_data_x.tolist()
# xgb_ext_pred_df['mol'] = test_data_x.tolist()
# xgb_ext_pred_df['true_AlogP'] = test_y.tolist()
# xgb_ext_pred_df.columns = ['pred_AlogP', 'mol', 'true_AlogP']
# xgb_ext_pred_df.to_csv('E:/tutorial/xgb_reg_ext_results.csv')

xgb_ext_df =  pd.DataFrame({'mol':test_data_x.tolist(),
                            'true_AlogP':test_y.tolist(),
                            'pred_AlogP':xgb_ext_pred})

xgb_ext_mae = mean_absolute_error(test_y, xgb_ext_pred)
xgb_ext_mse = mean_squared_error(test_y, xgb_ext_pred)
xgb_ext_mape = mean_absolute_percentage_error(test_y, xgb_ext_pred)
xgb_ext_r2 = r2_score(test_y, xgb_ext_pred)

xgb_perf = {'mae':[xgb_cv_test_mae, xgb_ext_mae],
            'mse':[xgb_cv_test_mse, xgb_ext_mse],
            'mape':[xgb_cv_test_mape, xgb_ext_mape],
            'r2':[xgb_cv_test_r2, xgb_ext_r2]}

xgb_perf_df = pd.DataFrame.from_dict(xgb_perf)
xgb_perf_df.index = ['cv', 'ext']
xgb_perf_df.to_csv('E:/tutorial/xgb_reg_perf_3.csv', index = True)

# 存储已经训练好的模型
# with open('E:/tutorial/xgb_reg.pkl', 'wb') as file:
    # pickle.dump(xgb_model, file)


# ext_dataset = pd.read_csv('E:/tutorial/toy_ext_set.csv')
# ext_mols = [Chem.MolFromSmiles(smi) for smi in ext_dataset['smiles']]
# ext_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4, 2048) for mol in ext_mols]
# ext_fps_array = np.asarray(ext_fps, dtype=float)

# ext_inputs = ext_fps_array
# ext_outputs = ext_dataset['mutagenicity']

# xgb_ext_pred_prob = xgb_model.predict_proba(ext_inputs)
# xgb_ext_pred_list = []
# for i, ext_score in enumerate(xgb_ext_pred_prob):
    # ext_score = ext_score[1]
    # xgb_ext_pred_list.append(ext_score)

# xgb_ext_pred_array = np.array(xgb_ext_pred_list)
# xgb_ext_auc = roc_auc_score(ext_outputs, xgb_ext_pred_list)
# xgb_ext_acc = accuracy_score(ext_outputs, np.round(xgb_ext_pred_array))
# xgb_ext_sen = recall_score(ext_outputs, np.round(xgb_ext_pred_array))
# xgb_ext_spc = (xgb_ext_acc * len(ext_outputs) - xgb_ext_sen * ext_outputs.sum())/(len(ext_outputs)-ext_outputs.sum())


# xgb_performance_dataset = {'AUC':[xgb_cv_test_auc, xgb_ext_auc],
                           # 'ACC':[xgb_cv_test_acc, xgb_ext_acc],
                           # 'SEN':[xgb_cv_test_sen, xgb_ext_sen],
                           # 'SPC':[xgb_cv_test_spc, xgb_ext_spc]}

# xgb_performance = pd.DataFrame(xgb_performance_dataset, index=['cv','ext'])
# xgb_performance.to_csv('E:/tutorial/xgb_performance.csv', index = False)

# with open('E:/tutorial/xgb.pkl', 'wb') as f:
    # pickle.dump(xgb_model, f)


# with open('E:\tutorial\xgb.pkl', 'rb') as xgb:
    # xgb = pickle.load(xgb)

# import pandas as pd
# import pickle
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import numpy as np

# with open('E:/tutorial/xgb.pkl', 'rb') as xgb:
    # xgb = pickle.load(xgb)

# data = pd.read_csv('E:/tutorial/ext.csv')
# mols = [Chem.MolFromSmiles(smi) for smi in data['SMILES']]
# morgan_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4, 2048) for mol in mols]
# morgan_fps_array = np.asarray(morgan_fps, dtype=float)

# xgb_ext_pred_prob = xgb.predict_proba(morgan_fps_array)


# smi = 'CC1CCN(S(=O)(=O)c2ccc3c(c2)C(=O)C(=O)C3Cc2ccccc2)CC(C)C1'
# mols = Chem.MolFromSmiles(smi)
# fp = [Chem.AllChem.GetMorganFingerprintAsBitVect(mols, 4, 2048)]
# fp_array = np.asarray(fp, dtype=float)
