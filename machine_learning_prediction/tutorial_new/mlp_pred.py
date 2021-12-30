import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

class QSAR_mlp(nn.Module):  # 继承torch的module，即可以直接拥有 torch.module 的属性和方法
    def __init__(self):
        super(QSAR_mlp, self).__init__()
        self.fc1 = nn.Linear(2048, 524) # 对输入数据进行线性转换
        self.fc2 = nn.Linear(524, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10,2) 
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        output = F.sigmoid(self.fc4(h3))
        return output

model = QSAR_mlp()
# model_dict=torch.load('E:/tutorial/model.pt')
model.load_state_dict(torch.load('E:/tutorial/model_dict.pt'))

ext_set = pd.read_csv('E:/tutorial/ext.csv')
ext_mols = [Chem.MolFromSmiles(smi) for smi in ext_set['SMILES']]
ext_morgan_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in ext_mols]
ext_morgan_fps_array = np.asarray(ext_morgan_fps, dtype=float)
ext_tensor = torch.from_numpy(ext_morgan_fps_array)

ext_pred = model(Variable(ext_tensor).float())
ext_predicted = torch.max(ext_pred, 1)[1]

