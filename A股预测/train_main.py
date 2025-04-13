# 首先 import 一些主要的包
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os


# 简单读出一个股票

# 获取文件名
file_name = 'train_data.npy'

# 读取数组
data = np.load(file_name)

# 简单展示信息
data
from sklearn.preprocessing import MinMaxScaler

# 这个 [0, 300] 是手动的预设值，可以自己更改
scaler = MinMaxScaler().fit(np.array([0, 300]).reshape(-1, 1))
# 生成题目所需的训练集合
def generate_data(data):

    # 记录 data 的长度
    n = data.shape[0]

    # 目标是生成可直接用于训练和测试的 x 和 y
    x = []
    y = []

    # 建立 (14 -> 1) 的 x 和 y
    for i in range(15, n):
        x.append(data[i-15:i-1])
        y.append(data[i-1])

    # 转换为 numpy 数组
    x = np.array(x)
    y = np.array(y)

    return x,y

x,y = generate_data(data)
print('x.shape : ', x.shape)
print('y.shape : ', y.shape)
# 生成 train valid test 集合，以供训练所需
def generate_training_data(x, y):
    # 样本总数
    num_samples = x.shape[0]
    # 测试集大小
    num_test = round(num_samples * 0.2)
    # 训练集大小
    num_train = round(num_samples * 0.7)
    # 校验集大小
    num_val = num_samples - num_test - num_train

    # 训练集拥有从 0 起长度为 num_train 的样本
    x_train, y_train = x[:num_train], y[:num_train]
    # 校验集拥有从 num_train 起长度为 num_val 的样本
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # 测试集拥有尾部 num_test 个样本
    x_test, y_test = x[-num_test:], y[-num_test:]

    # 返回这些集合
    return x_train, y_train, x_val, y_val, x_test, y_test

x_train, y_train, x_val, y_val, x_test, y_test = generate_training_data(x, y)
print('x_train.shape : ', x_train.shape)
print('y_train.shape : ', y_train.shape)
print('x_val.shape : ', x_val.shape)
print('y_val.shape : ', y_val.shape)
print('x_test.shape : ', x_test.shape)
print('y_test.shape : ', y_test.shape)
# 加载 pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# 获取数据中的 x, y
x,y = generate_data(data)

# 将 x,y 转换乘 tensor ， Pytorch 模型默认的类型是 float32
x = torch.tensor(x)
y = torch.tensor(y)

print(x.shape,y.shape)

# 将 y 转化形状
y = y.view(y.shape[0],1)

print(x.shape,y.shape)
# 对 x, y 进行 minmaxscale
x_scaled = scaler.transform(x.reshape(-1,1)).reshape(-1,14)
y_scaled = scaler.transform(y)

x_scaled = torch.tensor(x_scaled, dtype=torch.float32)
y_scaled = torch.tensor(y_scaled, dtype=torch.float32)
# 处理出训练集，校验集和测试集
x_train, y_train, x_val, y_val, x_test, y_test = generate_training_data(x_scaled, y_scaled)
# 建立一个自定 Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)
# 建立训练数据集、校验数据集和测试数据集
train_data = MyDataset(x_train,y_train)
valid_data = MyDataset(x_val,y_val)
test_data = MyDataset(x_test,y_test)

# 规定批次的大小
batch_size = 512

# 创建对应的 DataLoader
train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 校验集和测试集的 shuffle 是没有必要的，因为每次都会全部跑一遍
valid_iter = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_iter = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
for i, read_data in enumerate(test_iter):
    # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
    print("第 {} 个Batch \n{}".format(i, read_data))
    break
# 表示输出数据
print(read_data[0].shape, read_data[0])
# 表示输出标签
print(read_data[1].shape, read_data[1])

def train():
    '''训练模型
    :return: model 一个训练好的模型
    '''

    model = None
    # --------------------------- 此处下方加入训练模型相关代码 -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义 LSTM 模型（在 train() 内部）
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTMPredictor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # LSTM 层
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            
            # 全连接层
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    # 定义超参数
    input_size = 1       # 输入特征维度（假设只输入股票价格）
    hidden_size = 64     # LSTM 隐藏层大小
    num_layers = 2       # LSTM 层数
    output_size = 1      # 预测输出
    learning_rate = 0.001
    num_epochs = 80      # 训练轮数

    # 创建 LSTM 模型
    model = LSTMPredictor(input_size, hidden_size, num_layers, output_size).to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for features, labels in train_iter:  # 遍历训练数据

            features, labels = features.to(device), labels.to(device)

            # **修改：确保 features 形状为 (batch_size, sequence_length, input_size)**
            features = features.view(features.shape[0], -1, input_size)

            # **修改：确保 labels 形状与 outputs 匹配**
            labels = labels.view(-1, 1)

            optimizer.zero_grad()  # 清空梯度
            outputs = model(features)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_iter):.6f}")
        # print("Loss: {:.6f}".format(running_loss / len(train_iter)))
        
        
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # 在评估时不计算梯度，加速计算并节省显存
        for x_batch, y_batch in valid_iter:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # 确保数据在同一设备上
            # **修改：确保 x_batch 形状**
            x_batch = x_batch.view(x_batch.shape[0], -1, input_size)
            y_batch = y_batch.view(-1, 1)
            y_pred = model(x_batch)  # 进行预测
            loss = criterion(y_pred, y_batch)  # 计算损失
            total_loss += loss.item() * x_batch.size(0)  # 累加损失
            total_samples += x_batch.size(0)  # 统计样本数量

    avg_loss = total_loss / total_samples
    print(f"valid Loss: {avg_loss:.6f}")  # 只保留6位小数
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():  # 在评估时不计算梯度，加速计算并节省显存
        for x_batch, y_batch in test_iter:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # 确保数据在同一设备上
            # **修改：确保 x_batch 形状**
            x_batch = x_batch.view(x_batch.shape[0], -1, input_size)
            y_batch = y_batch.view(-1, 1)
            y_pred = model(x_batch)  # 进行预测
            loss = criterion(y_pred, y_batch)  # 计算损失
            total_loss += loss.item() * x_batch.size(0)  # 累加损失
            total_samples += x_batch.size(0)  # 统计样本数量

    avg_loss = total_loss / total_samples
    print(f"test Loss: {avg_loss:.6f}")  # 只保留6位小数
    
    




    # 如果使用的不是 pytorch 框架，还需要改动下面的代码
    # 模型保存的位置
    model_path = 'results/mymodel.pt'
    # 保存模型
    torch.save(model.state_dict(), model_path)
    # --------------------------- 此处上方加入训练模型相关代码 -------------------------------



    return model

train()
