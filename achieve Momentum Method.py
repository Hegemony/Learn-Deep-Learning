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


eta = 0.4  # 学习率

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

def show_trace_2d(f, results):
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')

# show_trace_2d(f_2d, d2l.train_2d(gd_2d))

# eta = 0.6
# show_trace_2d(f_2d, d2l.train_2d(gd_2d))

'''
可以看到使用较小的学习率η=0.4和动量超参数γ=0.5时，动量法在竖直方向上的移动更加平滑，且在水平方向上更快逼近最优解。
下面使用较大的学习率η=0.6，此时自变量也不再发散
'''
def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

eta, gamma = 0.4, 0.5
# show_trace_2d(f_2d, d2l.train_2d(momentum_2d))

eta = 0.6
# d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))

'''
从零开始实现：
相对于小批量随机梯度下降，动量法需要对每一个自变量维护一个同它一样形状的速度变量，且超参数里多了动量超参数。
实现中，我们将速度变量用更广义的状态变量states表示。
'''

features, labels = get_data_ch7()

def init_momentum_states():
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):  # params 和 states参数都分别为（w，b）和（v_w, v_b）
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data

'''
我们先将动量超参数momentum设0.5，这时可以看成是特殊的小批量随机梯度下降：其小批量随机梯度为最近2个时间步的2倍小批量梯度的加权平均。
'''
# d2l.train_ch7(sgd_momentum, init_momentum_states(),
#               {'lr': 0.02, 'momentum': 0.5}, features, labels)

'''
将动量超参数momentum增大到0.9，这时依然可以看成是特殊的小批量随机梯度下降：
其小批量随机梯度为最近10个时间步的10倍小批量梯度的加权平均。我们先保持学习率0.02不变。
'''
# d2l.train_ch7(sgd_momentum, init_momentum_states(),
#               {'lr': 0.02, 'momentum': 0.9}, features, labels)

'''
zip()用法
'''
# params = [1, 2, 3]
# states = [5, 6, 7]
# for p, v in zip(params, states):
#     print(p)
#     print('-'*100)
#     print(v)
# 1
# ----------------------------------------------------------------------------------------------------
# 5
# 2
# ----------------------------------------------------------------------------------------------------
# 6
# 3
# ----------------------------------------------------------------------------------------------------
# 7
#
# print('='*100)
# for p in zip(params, states):
#     print(p)
# (1, 5)
# (2, 6)
# (3, 7)
'''
简洁实现：
'''
d2l.train_pytorch_ch7(torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9},
                    features, labels)