# 导入需要的包
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# 调用模型文件
with open('E:/tutorial/rf.pkl', 'rb') as rf:
    rf = pickle.load(rf)

# 读取数据，计算分子指纹
data = pd.read_csv('E:/tutorial/ext.csv')
mols = [Chem.MolFromSmiles(smi) for smi in data['SMILES']]
morgan_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4, 2048) for mol in mols]
morgan_fps_array = np.asarray(morgan_fps, dtype=float)

#进行预测
rf_pred_prob = rf.predict_proba(morgan_fps_array)   # predict()

# 一个分子的预测
smi = 'CC1CCN(S(=O)(=O)c2ccc3c(c2)C(=O)C(=O)C3Cc2ccccc2)CC(C)C1'
mols = Chem.MolFromSmiles(smi)
fp = [Chem.AllChem.GetMorganFingerprintAsBitVect(mols, 4, 2048)]
fp_array = np.asarray(fp, dtype=float)
rf_pred_prob = rf.predict_proba(fp_array) # predict()
