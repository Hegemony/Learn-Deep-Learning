'''
torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
torchvision.utils: 其他的一些有用的方法。
'''
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
sys.path.append('F:/anaconda3/Lib/site-packages')  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST', train=True,
                                                download=True, transform=transforms.ToTensor())

mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST', train=False,
                                               download=True, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# feature, label = mnist_train[0]
# print(feature, feature.shape, label)

# 本函数已保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    # fig, ax = plt.subplots(1, 3, figsize = (15,7))，这样就会有1行3个15x7大小的子图。函数返回一个figure图像和子图ax的坐标系array列表。
    # print(figs)
    for f, img, lbl in zip(figs, images, labels):
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
        # print(f, lbl)
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)  # 不显示x轴
        f.axes.get_yaxis().set_visible(False)  # 不显示y轴
    plt.show()

X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
# print(X, y)
show_fashion_mnist(X, get_fashion_mnist_labels(y))

batch_size = 256
num_workers = 0  # 0表示不用额外的进程来加速读取数据

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)