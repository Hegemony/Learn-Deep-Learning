import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

# x = torch.rand(8, 1) * 20
# print(x)
'''
制作数据集
'''
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
#  numpy.random.normal(loc=0,scale=1e-2,size=shape) 正态分布，loc为均值，scale为标准差，size为大小
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
# 加入一些噪声

'''
读取数据
'''
batch_size = 10
# 将训练数据的样本和标签进行组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# for x, y in data_iter:
#     print(x, y)  # 打印一组
#     break
# print('-'*100)

'''
定义模型
'''
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    # 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net)

'''
打印模型参数
'''
for name, parameters in net.named_parameters():
    print(name, ':', parameters, '->', parameters.size())


'''
初始化模型参数
'''
# print(net.linear)
# print(net.linear.weight)
# print(net.linear.bias)
init.normal_(net.linear.weight, mean=0, std=0.01)  # 正态分布初始化，满足mean,std分布
init.constant_(net.linear.bias, val=0)  # 常数初始化，使值为常数val
# print(net.linear)
# print(net.linear.weight)
# print(net.linear.bias)

'''
定义损失函数
'''
loss = nn.MSELoss()

'''
定义优化算法
'''
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

'''
调整学习率
'''
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1  # 学习率为之前的0.1倍

'''
训练模型
'''
num_epochs = 10
for epoch in range(1, num_epochs+1):
    for X, y in data_iter:
        output = net(X)
        # print(output)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
    # print(l, l.item())  l返回的是tensor，l.item()返回的是该tensor的数值

'''
下面我们分别比较学到的模型参数和真实的模型参数。我们从net获得需要的层，并访问其权重（weight）和偏差（bias）。
学到的参数和真实的参数很接近。
'''
print(true_w, net.linear.weight)
print(true_b, net.linear.bias)