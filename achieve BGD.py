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
print(features.shape)  # torch.Size([1500, 5])
'''
genfromtxt的唯一强制参数是数据的源。它可以是字符串，字符串列表或生成器。如果提供了单个字符串，
则假定它是本地或远程文件或具有read方法的打开的类文件对象的名称，例如文件或StringIO.StringIO对象。
如果提供了字符串列表或返回字符串的生成器，则每个字符串在文件中被视为一行。当传递远程文件的URL时，文件将自动下载到当前目录并打开。 

识别的文件类型是文本文件和归档。目前，该函数识别gzip和bz2（bzip2）归档。归档的类型从文件的扩展名确定：
如果文件名以'.gz'结尾，则需要一个gzip归档；如果以'bz2'结尾，则假设存在一个bzip2档案
delimiter 参数:
一旦文件被定义并打开阅读，genfromtxt将每个非空行拆分为一个字符串序列。刚刚跳过空行或注释行。delimiter关键字用于定义拆分应如何进行。
通常，单个字符标记列之间的间隔。例如，逗号分隔文件（CSV）使用逗号（,）或分号（;）作为分隔符：

另一个常见的分隔符是"\t"，表格字符。但是，我们不限于单个字符，任何字符串都会做。默认情况下，genfromtxt假定delimiter=None，
表示该行沿白色空格（包括制表符）分割，并且连续的空格被视为单个白色空格。或者，我们可能处理固定宽度的文件，其中列被定义为给定数量的字符。
在这种情况下，我们需要将delimiter设置为单个整数（如果所有列具有相同的大小）或整数序列（如果列可以具有不同的大小）：
'''
'''
从零开始实现
'''
def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data

'''
下面实现一个通用的训练函数，以方便本章后面介绍的其他优化算法使用。它初始化一个线性回归模型，
然后可以使用小批量随机梯度下降以及后续小节介绍的其他算法来训练模型。
'''
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch7(optimizer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = d2l.linreg, d2l.squared_loss

    w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32),
                           requires_grad=True)
    b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()  # 使用平均损失

            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            l.backward()
            optimizer_fn([w, b], states, hyperparams)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # 每100个样本记录下当前训练误差
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    # d2l.set_figsize()
    # d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    # d2l.plt.xlabel('epoch')
    # d2l.plt.ylabel('loss')
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')

'''
当批量大小为样本总数1,500时，优化使用的是梯度下降。梯度下降的1个迭代周期对模型参数只迭代1次。
可以看到6次迭代后目标函数值（训练损失）的下降趋向了平稳。
'''
def train_sgd(lr, batch_size, num_epochs=2):
    train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)

# train_sgd(1, 1500, 6)

'''
当批量大小为1时，优化使用的是随机梯度下降。为了简化实现，有关（小批量）随机梯度下降的实验中，我们未对学习率进行自我衰减，
而是直接采用较小的常数学习率。随机梯度下降中，每处理一个样本会更新一次自变量（模型参数），一个迭代周期里会对自变量进行1,500次更新。
可以看到，目标函数值的下降在1个迭代周期后就变得较为平缓。
'''
# train_sgd(0.005, 1)

'''
虽然随机梯度下降和梯度下降在一个迭代周期里都处理了1,500个样本，但实验中随机梯度下降的一个迭代周期耗时更多。
这是因为随机梯度下降在一个迭代周期里做了更多次的自变量迭代，而且单样本的梯度计算难以有效利用矢量计算。
当批量大小为10时，优化使用的是小批量随机梯度下降。它在每个迭代周期的耗时介于梯度下降和随机梯度下降的耗时之间。
'''
# train_sgd(0.05, 10)

'''
简洁实现：
在PyTorch里可以通过创建optimizer实例来调用优化算法。这能让实现更简洁。下面实现一个通用的训练函数，
它通过优化算法的函数optimizer_fn和超参数optimizer_hyperparams来创建optimizer实例。
'''
# 本函数与原书不同的是这里第一个参数优化器函数而不是优化器的名字
# 例如: optimizer_fn=torch.optim.SGD, optimizer_hyperparams={"lr": 0.05}
def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    # 初始化模型
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1)
    )
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 除以2是为了和train_ch7保持一致, 因为squared_loss中除了2
            l = loss(net(X).view(-1), y) / 2

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    # d2l.set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')

train_pytorch_ch7(optim.SGD, {"lr": 0.05}, features, labels, 10)