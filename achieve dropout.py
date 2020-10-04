import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l

def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()

    return mask * X / keep_prob  # 做rescale

X = torch.arange(16).view(2, 8)
# print(X)
# print(torch.rand(X.shape))
# print(torch.rand(X.shape) < 0.5)
# print((torch.rand(X.shape) < 0.5).float())
# tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
#         [ 8,  9, 10, 11, 12, 13, 14, 15]])
# tensor([[0.5647, 0.1859, 0.8386, 0.4123, 0.0145, 0.1269, 0.5580, 0.5488],
#         [0.0491, 0.0197, 0.9416, 0.4735, 0.8994, 0.0464, 0.6942, 0.8963]])
# tensor([[False, False,  True,  True,  True,  True, False, False],
#         [False,  True,  True,  True,  True,  True, False, False]])
# tensor([[1., 0., 0., 0., 0., 1., 1., 0.],
#         [1., 0., 1., 0., 1., 0., 1., 0.]])

'''
dropout简介实现
'''

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)


drop_prob1, drop_prob2 = 0.2, 0.5

def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training:  # 只在训练模型时使用丢弃法
        H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层
    return torch.matmul(H2, W3) + b3

params = [W1, b1, W2, b2, W3, b3]
net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(drop_prob1),
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Dropout(drop_prob2),
        nn.Linear(num_hiddens2, 10)
        )

for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

# 在PyTorch中，我们只需要在全连接层后添加Dropout层并指定丢弃概率。
# 在训练模型时，Dropout层将以指定的丢弃概率随机丢弃上一层的输出元素；
# 在测试模型时（即model.eval()后），Dropout层并不发挥作用。
