import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
DenseNet的主要构建模块是稠密块（dense block）和过渡层（transition layer）。
前者定义了输入和输出是如何连结的，后者则用来控制通道数，使之不过大。
'''

'''
稠密块：
DenseNet使用了ResNet改良版的“批量归一化、激活和卷积”结构，我们首先在conv_block函数里实现这个结构。
'''
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

'''
稠密块由多个conv_block组成，每块使用相同的输出通道数。但在前向计算时，我们将每块的输入和输出在通道维上连结。
'''
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels  # 计算输出通道数,供下一个过渡块使用

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X

'''
我们定义一个有2个输出通道数为10的卷积块。使用通道数为3的输入时，我们会得到通道数为3+2×10=23的输出。
卷积块的通道数控制了输出通道数相对于输入通道数的增长，因此也被称为增长率（growth rate）
'''
blk = DenseBlock(2, 3, 10)  # 给网络模型赋予参数
X = torch.rand(4, 3, 8, 8)   # 输入数据前向传播
Y = blk(X)
print(Y.shape)  # torch.Size([4, 23, 8, 8])

'''
过渡层：
由于每个稠密块都会带来通道数的增加，使用过多则会带来过于复杂的模型。过渡层用来控制模型复杂度。
它通过1×1卷积层来减小通道数，并使用步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。
'''
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
    return blk

'''
对上一个例子中稠密块的输出使用通道数为10的过渡层。此时输出的通道数减为10，高和宽均减半。
'''
blk = transition_block(23, 10)   # 给网络模型赋予参数
print(blk(Y).shape)   # torch.Size([4, 10, 4, 4])

'''
DenseNet模型:
我们来构造DenseNet模型。DenseNet首先使用同ResNet一样的单卷积层和最大池化层。
'''
net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

'''
类似于ResNet接下来使用的4个残差块，DenseNet使用的是4个稠密块。同ResNet一样，我们可以设置每个稠密块使用多少个卷积层。
这里我们设成4，从而与上一节的ResNet-18保持一致。稠密块里的卷积层通道数（即增长率）设为32，所以每个稠密块将增加128个通道。
ResNet里通过步幅为2的残差块在每个模块之间减小高和宽。这里我们则使用过渡层来减半高和宽，并减半通道数。
'''
num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):  # 给网络模型赋予参数
    DB = DenseBlock(num_convs, num_channels, growth_rate)
    net.add_module("DenseBlosk_%d" % i, DB)
    # 上一个稠密块的输出通道数
    num_channels = DB.out_channels
    # 在稠密块之间加入通道数减半的过渡层
    if i != len(num_convs_in_dense_blocks) - 1:
        net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

'''
同ResNet一样，最后接上全局池化层和全连接层来输出
'''
X = torch.rand((1, 1, 96, 96))
print(net)
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)
