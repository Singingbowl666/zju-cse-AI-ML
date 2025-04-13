# Copyright (c) 2025.3 Hanzhe Ma 3220105872@zju.edu.cn
# All rights reserved.

# 1.导入相关第三方库或者包（根据自己需求，可以增加、删除等改动）
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# 2.导入 Notebook 使用的模型
# 建立一个简单的线性模型
class LinearNet(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        # 一个线性层
        self.linear = torch.nn.Linear(num_inputs, num_outputs)

    # 前向传播函数
    def forward(self, x): # x shape: (batch, 14)
        y = self.linear(x)
        return y


# 加载 Notebook 模型流程

# 输入的数量是前 14 个交易日的收盘价
num_inputs = 14
# 输出是下一个交易日的收盘价
num_outputs = 1

# ------------------------- 请加载您最满意的模型网络结构 -----------------------------
# 读取模型
#model = LinearNet(num_inputs,num_outputs)
# 选择 CPU / GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层（取最后时间步的输出）
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 定义 LSTM 超参数（确保与你训练时的超参数一致）
input_size = 1       # 每天的输入是 1 维特征（股票价格）
hidden_size = 64     # LSTM 隐藏层大小
num_layers = 2       # LSTM 层数
output_size = 1      # 预测输出
model = LSTMPredictor(input_size, hidden_size, num_layers, output_size).to(device)

# ----------------------------- 请加载您最满意的模型 -------------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 temp.pth 模型，则 model_path = 'results/mymodel.pt'
# 模型保存的位置，如果模型路径不同，请修改！！！
model_path = 'results/mymodel.pt'
#model.load_state_dict(torch.load(model_path))
# 加载模型, 确保在 CPU 设备上运行
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.eval()

def predict(test_x):
    '''
    对于给定的 x 预测未来的 y 。
    :param test_x: 给定的数据集合 x ，对于其中的每一个元素需要预测对应的 y 。e.g.:np.array([[6.69,6.72,6.52,6.66,6.74,6.55,6.35,6.14,6.18,6.17,5.72,5.78,5.69,5.67]]
    :return: test_y 对于每一个 test_x 中的元素，给出一个对应的预测值。e.g.:np.array([[0.0063614]])
    '''
    # test 的数目
    n_test = test_x.shape[0]

    test_y = None
    # --------------------------- 此处下方加入读入模型和预测相关代码 -------------------------------
    # 此处为 Notebook 模型示范，你可以根据自己数据处理方式进行改动
    scaler = MinMaxScaler().fit(np.array([0, 300]).reshape(-1, 1))
    test_x = scaler.transform(test_x.reshape(-1, 1)).reshape(-1, 14)
    
    test_x = test_x.reshape(n_test, 14, 1)
    test_x = torch.tensor(test_x, dtype=torch.float32)

    test_y = model(test_x)

    # 如果使用 MinMaxScaler 进行数据处理，预测后应使用下一句将预测值放缩到原范围内
    test_y = scaler.inverse_transform(test_y.detach().cpu())
    if isinstance(test_y, torch.Tensor):
        test_y = test_y.detach().cpu().numpy()
    # --------------------------- 此处上方加入读入模型和预测相关代码 -------------------------------

    # 保证输出的是一个 numpy 数组
    assert(type(test_y) == np.ndarray)

    # 保证 test_y 的 shape 正确
    assert(test_y.shape == (n_test, 1))

    return test_y
