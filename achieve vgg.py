import torch
import time
from torch import nn, optim

import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
VGG块
'''
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 这里会使宽高减半

    return nn.Sequential(*blk)

'''
VGG网络：
与AlexNet和LeNet一样，VGG网络由卷积层模块后接全连接层模块构成。卷积层模块串联数个vgg_block，
其超参数由变量conv_arch定义。该变量指定了每个VGG块里卷积层个数和输入输出通道数。全连接模块则跟AlexNet中的一样。
'''
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7   # c * w * h
fc_hidden_units = 4096  # 任意

'''
现在我们构造一个VGG网络。它有5个卷积块，前2块使用单卷积层，而后3块使用双卷积层。
第一块的输入输出通道分别是1（因为下面要使用的Fashion-MNIST数据的通道数为1）和64，之后每次对输出通道数翻倍，
直到变为512。因为这个网络使用了8个卷积层和3个全连接层，所以经常被称为VGG-11。
'''
def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))  # 将一个child module 添加到当前modle, 被添加的module可以通过name属性来获取，
    # 全连接层部分
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 10)
                                ))
    return net


'''
下面构造一个高和宽均为224的单通道数据样本来观察每一层的输出形状。
'''
net = vgg(conv_arch, fc_features, fc_hidden_units)
print(net)  # 输出网络结构
print('-'*100)
X = torch.rand(1, 1, 224, 224)

# named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
# named_children( ):
# 返回包含子模块的迭代器，同时产生模块的名称以及模块本身。
# named_modules( ):
# 返回网络中所有模块的迭代器，同时产生模块的名称以及模块本身。
for name, blk in net.named_children():
    print(blk)
    X = blk(X)
    print(name, 'output shape: ', X.shape)
    print('-'*100)

