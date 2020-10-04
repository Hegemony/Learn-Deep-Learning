'''
权重衰减等价于 L2范数正则化（regularization）。正则化通过为模型损失函数添加惩罚项使学出的模型参数值较小，是
应对过拟合的常用手段。我们先描述L2范数正则化，再解释它为何又称权重衰减。
'''
import torch
import torch.nn as nn
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch import optim
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l

'''
下面，我们以高维线性回归为例来引入一个过拟合问题，并使用权重衰减来应对过拟合。设数据样本特征的维度为p。
对于训练数据集和测试数据集中特征为x1,x2,…,xp的任一样本，我们使用如下的线性函数来生成该样本的标签：
y=0.05+∑pi=10.01xi+ϵ
其中噪声项ϵϵ服从均值为0、标准差为0.01的正态分布。为了较容易地观察过拟合，我们考虑高维线性回归问题，
如设维度p=200；同时，我们特意把训练数据集的样本数设低，如20。
'''
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]
'''
从零开始实现:
下面先介绍从零开始实现权重衰减的方法。我们通过在目标函数后添加L2范数惩罚项来实现权重衰减
'''
'''
初始化模型参数:
首先，定义随机初始化模型参数的函数。该函数为每个参数都附上梯度。
'''
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


'''
定义L2范数惩罚项:
下面定义L2范数惩罚项。这里只惩罚模型的权重参数。
'''
def l2_penalty(w):
    return (w**2).sum() / 2


'''
定义训练和测试:
下面定义如何在训练数据集和测试数据集上分别训练和测试模型。与前面几节中不同的是，这里在计算最终的损失函数时添加了L2范数惩罚项。
'''
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def fit_and_plot(lambd):
    w, b = init_params()
    optimizer = optim.SGD([w, b], lr=lr)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了L2范数惩罚项
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            # d2l.sgd([w, b], lr, batch_size)
            optimizer.step()
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    # plt.semilogx函数，其中常用的是semilogy函数，即后标为x的是在x轴取对数，为y的是y轴坐标取对数。loglog是x y轴都取对数。
    print('L2 norm of w:', w.norm().item())


'''
观察过拟合:
接下来，让我们训练并测试高维线性回归模型。当lambd设为0时，我们没有使用权重衰减。结果训练误差远小于测试集上的误差。这是典型的过拟合现象。
'''
# fit_and_plot(lambd=0)

'''
使用权重衰减:
下面我们使用权重衰减。可以看出，训练误差虽然有所提高，但测试集上的误差有所下降。过拟合现象得到一定程度的缓解。另外，权重参数的L2
范数比不使用权重衰减时的更小，此时的权重参数更接近0。
'''
fit_and_plot(lambd=3)

'''
正则化通过为模型损失函数添加惩罚项使学出的模型参数值较小，是应对过拟合的常用手段。
权重衰减等价于L2范数正则化，通常会使学到的权重参数的元素较接近0。
权重衰减可以通过优化器中的weight_decay超参数来指定。
可以定义多个优化器实例对不同的模型参数使用不同的迭代方法。
'''