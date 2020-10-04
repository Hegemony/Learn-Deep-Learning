import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l

def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x
        results.append(x)
    print('epoch 10 x:', x)
    return results

res = gd(0.2)
print(res)
'''
下面将绘制出自变量x的迭代轨迹。
'''
def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    ax = plt.figure()
    fig = ax.add_subplot(111)
    fig.plot(f_line, [x * x for x in f_line])
    fig.plot(res, [x * x for x in res], '-o')
    plt.xlabel('x')
    plt.ylabel('f(x)')

# show_trace(res)

'''
学习率：
上述梯度下降算法中的正数η通常叫作学习率。这是一个超参数，需要人工设定。如果使用过小的学习率，
会导致x更新缓慢从而需要更多的迭代才能得到较好的解。
'''
# show_trace(gd(0.05))  # 学习率过小
# show_trace(gd(1.1))   # 学习率过大

'''
多维度梯度下降
'''
'''
下面我们构造一个输入为二维向量x=[x1,x2]⊤和输出为标量的目标函数f(x)=x1^2+2x2^2。
那么，梯度∇f(x)=[2x1,4x2]⊤。我们将观察梯度下降从初始位置[−5,−2]开始对自变量xx的迭代轨迹。
我们先定义两个辅助函数，第一个函数使用给定的自变量更新函数，从初始位置[−5,−2]开始迭代自变量x共20次，
第二个函数对自变量x的迭代轨迹进行可视化。
'''
def train_2d(trainer):  # 本函数将保存在d2lzh_pytorch包中方便以后使用
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1和s2是自变量状态，本章后续几节会使用
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results

def show_trace_2d(f, results):  # 本函数将保存在d2lzh_pytorch包中方便以后使用
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')  # 绘制等高线
    plt.xlabel('x1')
    plt.ylabel('x2')

eta = 0.1

def f_2d(x1, x2):  # 目标函数
    return x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):  # 求梯度函数
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)

# show_trace_2d(f_2d, train_2d(gd_2d))  # 函数名当作参数名

# print(train_2d(gd_2d))
# print(*zip(train_2d(gd_2d)))
# print(*zip(*train_2d(gd_2d)))
# zip()：压缩, zip(a,b)zip()函数分别从a和b依次各取出一个元素组成元组，再将依次组成的元组组合成一个新的迭代器--新的zip类型数据。
# *zip()：解压, *zip()函数是zip()函数的逆过程，将zip对象变成原先组合前的数据。
# 参数前面加上* 号 ，意味着参数的个数不止一个，另外带一个星号（*）参数的函数传入的参数存储为一个元组（tuple），
# 带两个（*）号则是表示字典（dict）

'''
[X,Y] = meshgrid(x,y) 将向量x和y定义的区域转换成矩阵X和Y,其中矩阵X的行向量是向量x的简单复制，
而矩阵Y的列向量是向量y的简单复制(注：下面代码中X和Y均是数组，在文中统一称为矩阵了)。
X是按照行复制，Y是按照列复制。
假设x是长度为m的向量，y是长度为n的向量，则最终生成的矩阵X和Y的维度都是(n,m) （注意不是(m,n)）。
'''
# x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
# print(x1, x2)
# print(x1.shape, x2.shape)
# [[-5.5 -5.4 -5.3 ...  0.7  0.8  0.9]
#  [-5.5 -5.4 -5.3 ...  0.7  0.8  0.9]
#  [-5.5 -5.4 -5.3 ...  0.7  0.8  0.9]
#  ...
#  [-5.5 -5.4 -5.3 ...  0.7  0.8  0.9]
#  [-5.5 -5.4 -5.3 ...  0.7  0.8  0.9]
#  [-5.5 -5.4 -5.3 ...  0.7  0.8  0.9]] [[-3.  -3.  -3.  ... -3.  -3.  -3. ]
#  [-2.9 -2.9 -2.9 ... -2.9 -2.9 -2.9]
#  [-2.8 -2.8 -2.8 ... -2.8 -2.8 -2.8]
#  ...
#  [ 0.7  0.7  0.7 ...  0.7  0.7  0.7]
#  [ 0.8  0.8  0.8 ...  0.8  0.8  0.8]
#  [ 0.9  0.9  0.9 ...  0.9  0.9  0.9]]
# (40, 65) (40, 65)

'''
随机梯度下降：
通过在梯度中添加均值为0的随机噪声来模拟随机梯度下降。
'''
def sgd_2d(x1, x2, s1, s2):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
            x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)

show_trace_2d(f_2d, train_2d(sgd_2d))
'''
可以看到，随机梯度下降中自变量的迭代轨迹相对于梯度下降中的来说更为曲折。这是由于实验所添加的噪声使模拟的随机梯度的准确度下降。
在实际中，这些噪声通常指训练数据集中的无意义的干扰。
'''

'''
使用适当的学习率，沿着梯度反方向更新自变量可能降低目标函数值。梯度下降重复这一更新过程直到得到满足要求的解。
学习率过大或过小都有问题。一个合适的学习率通常是需要通过多次实验找到的。
当训练数据集的样本较多时，梯度下降每次迭代的计算开销较大，因而随机梯度下降通常更受青睐。
'''