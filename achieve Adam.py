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

features, labels = get_data_ch7()

'''
从零开始实现:
我们按照Adam算法中的公式实现该算法。其中时间步tt通过hyperparams参数传入adam函数
'''
def init_adam_states():
    v_w, v_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad.data
        s[:] = beta2 * s + (1 - beta2) * p.grad.data**2
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p.data -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1


'''
使用学习率为0.01的Adam算法来训练模型。
'''
# d2l.train_ch7(adam, init_adam_states(), {'lr': 0.01, 't': 1}, features, labels)

'''
简洁实现:
通过名称为“Adam”的优化器实例，我们便可使用PyTorch提供的Adam算法。
'''
d2l.train_pytorch_ch7(torch.optim.Adam, {'lr': 0.01}, features, labels)

'''
Adam算法在RMSProp算法的基础上对小批量随机梯度也做了指数加权移动平均。
Adam算法使用了偏差修正。
'''