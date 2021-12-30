import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import pickle

dataset = pd.read_csv('E:/tutorial/toy_set.csv')

mols = [Chem.MolFromSmiles(smi) for smi in dataset['smiles']]
morgan_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4, 2048) for mol in mols]
morgan_fps_array = np.asarray(morgan_fps, dtype=float)

inputs = morgan_fps_array
outputs = dataset['mutagenicity']

scoring = {'AUC':'roc_auc',
           'ACC':make_scorer(accuracy_score),
           'SEN':make_scorer(recall_score)}

xgb_param_grid = {'n_estimators ':[2, 4],
                  'max_depth':[1, 2, 3]}

print('Grid search of XGBoost...')
xgb_classifier = xgb.XGBClassifier()
xgb_gs = GridSearchCV(xgb_classifier,
                      xgb_param_grid,
                      scoring = scoring,
                      cv = 5,
                      n_jobs = 12,
                      refit = 'AUC',
                      return_train_score = True)

xgb_gs_ecfp = xgb_gs.fit(inputs, outputs)
xgb_model = xgb_gs_ecfp.best_estimator_

print('Cross Validation of XGBoost...')
xgb_cv = cross_validate(xgb_model,
                        inputs,
                        outputs,
                        cv = 5,
                        n_jobs = 12,
                        scoring = scoring,
                        return_train_score = True)

print('Calculate metrics of XGBoost cross validation...')
xgb_cv_train_auc = np.mean(xgb_cv['train_AUC'])
xgb_cv_test_auc = np.mean(xgb_cv['test_AUC'])
xgb_cv_train_acc = np.mean(xgb_cv['train_ACC'])
xgb_cv_test_acc = np.mean(xgb_cv['test_ACC'])
xgb_cv_train_sen = np.mean(xgb_cv['train_SEN'])
xgb_cv_test_sen = np.mean(xgb_cv['test_SEN'])
xgb_cv_train_spc = (xgb_cv_train_acc * len(outputs) - xgb_cv_train_sen * outputs.sum())/(len(outputs)-outputs.sum())
xgb_cv_test_spc = (xgb_cv_test_acc * len(outputs) - xgb_cv_test_sen * outputs.sum())/(len(outputs)-outputs.sum())

ext_dataset = pd.read_csv('E:/tutorial/toy_ext_set.csv')
ext_mols = [Chem.MolFromSmiles(smi) for smi in ext_dataset['smiles']]
ext_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4, 2048) for mol in ext_mols]
ext_fps_array = np.asarray(ext_fps, dtype=float)

ext_inputs = ext_fps_array
ext_outputs = ext_dataset['mutagenicity']

xgb_ext_pred_prob = xgb_model.predict_proba(ext_inputs)
xgb_ext_pred_list = []
for i, ext_score in enumerate(xgb_ext_pred_prob):
    ext_score = ext_score[1]
    xgb_ext_pred_list.append(ext_score)

xgb_ext_pred_array = np.array(xgb_ext_pred_list)
xgb_ext_auc = roc_auc_score(ext_outputs, xgb_ext_pred_list)
xgb_ext_acc = accuracy_score(ext_outputs, np.round(xgb_ext_pred_array))
xgb_ext_sen = recall_score(ext_outputs, np.round(xgb_ext_pred_array))
xgb_ext_spc = (xgb_ext_acc * len(ext_outputs) - xgb_ext_sen * ext_outputs.sum())/(len(ext_outputs)-ext_outputs.sum())


xgb_performance_dataset = {'AUC':[xgb_cv_test_auc, xgb_ext_auc],
                           'ACC':[xgb_cv_test_acc, xgb_ext_acc],
                           'SEN':[xgb_cv_test_sen, xgb_ext_sen],
                           'SPC':[xgb_cv_test_spc, xgb_ext_spc]}

xgb_performance = pd.DataFrame(xgb_performance_dataset, index=['cv','ext'])
xgb_performance.to_csv('E:/tutorial/xgb_performance.csv', index = False)

with open('E:/tutorial/xgb.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)


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
