import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
常用的图像增广方法:
我们来读取一张形状为400×500（高和宽分别为400像素和500像素）的图像作为实验的样例。
'''
img = Image.open('./img/cat1.jpg')
# plt.imshow(img)

'''
下面定义绘图函数show_images
'''
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    # print(a)
    # print(axes)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)  # x 轴不可见
            axes[i][j].axes.get_yaxis().set_visible(False)  # y 轴不可见
    return axes

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)

'''
翻转和裁剪:
左右翻转图像通常不改变物体的类别。它是最早也是最广泛使用的一种图像增广方法。
下面我们通过torchvision.transforms模块创建RandomHorizontalFlip实例来实现一半概率的图像水平（左右）翻转。
'''
# apply(img, torchvision.transforms.RandomHorizontalFlip())

'''
上下翻转不如左右翻转通用。但是至少对于样例图像，上下翻转不会造成识别障碍。
下面我们创建RandomVerticalFlip实例来实现一半概率的图像垂直（上下）翻转。
'''
# apply(img, torchvision.transforms.RandomVerticalFlip())

'''
在下面的代码里，我们每次随机裁剪出一块面积为原面积10%∼100%的区域，且该区域的宽和高之比随机取自0.5∼2，
然后再将该区域的宽和高分别缩放到200像素。若无特殊说明，本节中a和b之间的随机数指的是从区间[a,b]中随机均匀采样所得到的连续值。
'''
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
# class torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
# 功能：随机大小，随机长宽裁剪原始照片，最后将照片resize到设定好的size
# 参数：
# size：输出的分辨率，就是输出的大小
# scale：随机剪裁的大小区间，上体来说，crop出来的图片会在0.08倍到1倍之间
# ratio：随机宽长比设置
# interpolation：插值的方法。
print(shape_aug)
# apply(img, shape_aug)

'''
变化颜色:
另一类增广方法是变化颜色。我们可以从4个方面改变图像的颜色：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。
在下面的例子里，我们将图像的亮度随机变化为原图亮度的50%（1−0.5）∼150%（1+0.5）。
'''
# apply(img, torchvision.transforms.ColorJitter(brightness=0.5))

'''
我们也可以随机变化图像的色调。
'''
# apply(img, torchvision.transforms.ColorJitter(hue=0.5))

'''
类似地，我们也可以随机变化图像的对比度。
'''
# apply(img, torchvision.transforms.ColorJitter(contrast=0.5))

'''
我们也可以同时设置如何随机变化图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。
'''
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)

# transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
# 功能：调整亮度、对比度、饱和度和色相
# brightness：亮度调整因子
# 当为a时，从[max(0, 1-a), 1+a]中随机选择
# 当为(a, b)时，从[a, b]中
# contrast：对比度参数，同brightness
# saturation：饱和度参数，同brightness
# hue：色相参数，当为a时，从[-a, a]中选择参数，注： 0<= a <= 0.5
# 当为(a, b)时，从[a, b]中选择参数，注：-0.5 <= a <= b <= 0.5
'''
叠加多个图像增广方法:
实际应用中我们会将多个图像增广方法叠加使用。我们可以通过Compose实例将上面定义的多个图像增广方法叠加起来，再应用到每张图像之上。
'''
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    color_aug,
    shape_aug]
    )
apply(img, augs)

'''
使用图像增广训练模型:
下面我们来看一个将图像增广应用在实际训练中的例子。这里我们使用CIFAR-10数据集，而不是之前我们一直使用的Fashion-MNIST数据集。
这是因为Fashion-MNIST数据集中物体的位置和尺寸都已经经过归一化处理，而CIFAR-10数据集中物体的颜色和大小区别更加显著。
下面展示了CIFAR-10数据集中前32张训练图像。
'''
all_imges = torchvision.datasets.CIFAR10(train=True, root="./Datasets/CIFAR-10", download=True)
# all_imges的每一个元素都是(image, label)
show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=0.8)

'''
为了在预测时得到确定的结果，我们通常只将图像增广应用在训练样本上，而不在预测时使用含随机操作的图像增广。
在这里我们只使用最简单的随机左右翻转。此外，我们使用ToTensor将小批量图像转成PyTorch需要的格式，
即形状为(批量大小, 通道数, 高, 宽)、值域在0到1之间且类型为32位浮点数。
'''
flip_aug = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()]
    )

no_aug = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()]
    )

'''
接下来我们定义一个辅助函数来方便读取图像并应用图像增广。有关DataLoader的详细介绍，可参考更早的3.5节图像分类数据集(Fashion-MNIST)。
'''
num_workers = 0
def load_cifar10(is_train, augs, batch_size, root="./Datasets/CIFAR-10"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

'''
使用图像增广训练模型
'''
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            # print(y_hat, y_hat.size())  # 256*10
            # print('-'*100)
            # print(y, y.shape)   # 256
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            # argmax(dim=1) 返回每行最大值的索引，(y_hat.argmax(dim=1) == y) ->[True, False,....], sum()->True+False=1
            n += y.shape[0]
            batch_count += 1
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

'''
然后就可以定义train_with_data_aug函数使用图像增广来训练模型了。该函数使用Adam算法作为训练使用的优化算法，
然后将图像增广应用于训练数据集之上，最后调用刚才定义的train函数训练并评价模型
'''
def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, d2l.resnet18(10)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)

'''
下面使用随机左右翻转的图像增广来训练模型。
'''
train_with_data_aug(flip_aug, no_aug)

# a = torch.tensor([1, 2, 3])
# b = torch.tensor((1, 2, 2))
# print((a == b))  # tensor([ True,  True, False])
# print((a == b).sum())  # tensor(2)

'''
图像增广基于现有训练数据生成随机图像从而应对过拟合。
为了在预测时得到确定的结果，通常只将图像增广应用在训练样本上，而不在预测时使用含随机操作的图像增广。
可以从torchvision的transforms模块中获取有关图片增广的类。
'''