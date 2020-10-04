import numpy as np
import torch
import time
from torch import nn, optim
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l

def get_data_ch7():  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    data = np.genfromtxt('./data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    # print(data.shape)  # 1503*5
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
           torch.tensor(data[:1500, -1], dtype=torch.float32)  # 前1500个样本(每个样本5个特征)


'''
从零开始实现:
AdaDelta算法需要对每个自变量维护两个状态变量，即st和Δxt。我们按AdaDelta算法中的公式实现该算法。
'''
features, labels = get_data_ch7()

def init_adadelta_states():
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    delta_w, delta_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):  # 依次取出 w, b 进行计算
        s[:] = rho * s + (1 - rho) * (p.grad.data**2)
        # print(s)
        g = p.grad.data * torch.sqrt((delta + eps) / (s + eps))
        p.data -= g
        delta[:] = rho * delta + (1 - rho) * g * g

'''
使用超参数ρ=0.9来训练模型。
'''
# d2l.train_ch7(adadelta, init_adadelta_states(), {'rho': 0.9}, features, labels)

'''
简洁实现:
通过名称为Adadelta的优化器方法，我们便可使用PyTorch提供的AdaDelta算法。它的超参数可以通过rho来指定。
'''
d2l.train_pytorch_ch7(torch.optim.Adadelta, {'rho': 0.9}, features, labels)

'''
AdaDelta算法没有学习率超参数，它通过使用有关自变量更新量平方的指数加权移动平均的项来替代RMSProp算法中的学习率。
'''