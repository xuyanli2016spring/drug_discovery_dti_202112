# 导入包
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

# 读取数据
train_set = pd.read_csv('E:/tutorial/toy_set.csv')

# 设置参数
epochs = 10

# 计算分子指纹
train_mols = [Chem.MolFromSmiles(smi) for smi in train_set['smiles']]
train_morgan_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in train_mols]
train_morgan_fps_array = np.asarray(train_morgan_fps, dtype=float)
train_class = train_set['mutagenicity'] # 提取数据标签
train_x, test_x, train_y, test_y = train_test_split(train_morgan_fps_array, train_class, test_size=0.2, random_state=0) # 切分数据

# 数据形式转换（从数组（array）到张量（tensor））
train_x_tensor = torch.from_numpy(train_x)
test_x_tensor = torch.from_numpy(test_x)
train_y_tensor = torch.from_numpy(np.array(train_y))
test_y_tensor = torch.from_numpy(np.array(test_y))

print(train_x_tensor.size(),train_y_tensor.size())
print(test_x_tensor.size(), test_y_tensor.size())


class QSAR_mlp(nn.Module):  # 继承torch的module，即可以直接拥有 torch.nn.Module 的属性和方法
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
print(model)

losses = []
optimizer = optim.Adam(model.parameters(), lr=0.005)
for epoch in range(20):
    data, target = Variable(train_x_tensor).float(), Variable(train_y_tensor).long()
    optimizer.zero_grad()
    y_pred = model(data)
    loss = F.cross_entropy(y_pred, target)
    losses.append(loss.item())
    print("Loss: {}".format(loss.item()))
    loss.backward()
    optimizer.step()

pred_y = model(Variable(test_x_tensor).float())
predicted = torch.max(pred_y, 1)[1]

for i in range(len(predicted)):
    print("pred:{}, target:{}".format(predicted.data[i], test_y_tensor[i]))

def confusion_matrix(prediction, truth):
    confusion_vector = prediction / truth
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()
    return tp, fp, tn, fn

tp, fp, tn, fn = confusion_matrix(predicted.data, test_y_tensor)

recall = tp / (tp+fn)
precision = tp / (tp+fp)
accuracy = (tp+tn) / (tp+fp+tn+fn)

torch.save(model, 'E:/tutorial/model.pt') # 保存整个网络
torch.save(model.state_dict(), 'E:/tutorial/model_dict.pt') # 保存网络参数，速度快，占空间少

# pytorch 模型调用
# model_dict=torch.load('E:/tutorial/model.pt')
# model_dict=model.load_state_dict(torch.load('E:/tutorial/model_dict.pt'))