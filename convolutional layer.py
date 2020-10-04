import torch
from torch import nn
import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l

# 它接受输入数组X与核数组K，并输出数组Y，手动实现卷积运算
def corr2d(X, K):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))
print('-'*100)

'''
二维卷积层
'''
# 基于上面的corr2d函数来实现一个自定义的二维卷积层。
# 在构造函数__init__里我们声明weight和bias这两个模型参数。
# 前向计算函数forward则是直接调用corr2d函数再加上偏差。
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


'''
图像中的物体边缘检测
'''
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1, -1]])
# 构造一个高和宽分别为1和2的卷积核K。当它与输入做互相关运算时，如果横向相邻元素相同，输出为0；否则输出为非0
Y = corr2d(X, K)
print(Y)
print('-'*100)


'''
通过数据学习核数组
'''
# 构造一个核数组形状是(1, 2)的二维卷积层
conv2d = Conv2D(kernel_size=(1, 2))

step = 20
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清0
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))

print("weight: ", conv2d.weight.data)
print("bias:", conv2d.bias.data)
#  学到的卷积核的权重参数与我们之前定义的核数组K较接近，而偏置参数接近0

K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
a = torch.stack([K, K+1, K+2])
print(a)
print('-'*100)

'''
卷积操作多输入通道
'''
def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道维）分别计算再相加
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])  # 卷积的结果进行累加
    return res
X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(corr2d_multi_in(X, K))

'''
卷积操作多输出通道
'''
def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])
K = torch.stack([K, K + 1, K + 2])
print(K, K.shape)
print(corr2d_multi_in_out(X, K))
