import torch
from torch import nn

def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(pool2d(X, (2, 2)))   # 最大池化
print(pool2d(X, (2, 2), 'avg'))  # 平均池化

'''
填充和步幅
'''
X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
print(X)
pool2d = nn.MaxPool2d(3)
# 参数：
# kernel_size(int or tuple) - max pooling的窗口大小，
# stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
# padding(int or tuple, optional) - 输入的每一条边补充0的层数
# dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
# return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
# ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作
print(pool2d(X))

'''
我们也可以指定非正方形的池化窗口，并分别指定高和宽上的填充和步幅。
'''
pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
print(pool2d(X))

'''
多通道
'''
X = torch.cat((X, X + 1), dim=1)
print(X)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
