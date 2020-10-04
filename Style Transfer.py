import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
读取内容图像和样式图像:
首先，我们分别读取内容图像和样式图像。从打印出的图像坐标轴可以看出，它们的尺寸并不一样。
'''
# d2l.set_figsize()
content_img = Image.open('./Data/rainier.jpg')
# d2l.plt.imshow(content_img)
# plt.imshow(content_img)

# d2l.set_figsize()
style_img = Image.open('./Data/autumn_oak.jpg')
# d2l.plt.imshow(style_img)
# plt.imshow(style_img)

'''
预处理和后处理图像:
下面定义图像的预处理函数和后处理函数。预处理函数preprocess对先对更改输入图像的尺寸，然后再将PIL图片转成卷积神经网络接受的输入格式，
再在RGB三个通道分别做标准化，由于预训练模型是在均值为[0.485, 0.456, 0.406]标准差为[0.229, 0.224, 0.225]的图片数据上预训练的，
所以我们要将图片标准化保持相同的均值和标准差。后处理函数postprocess则将输出图像中的像素值还原回标准化之前的值。
由于图像每个像素的浮点数值在0到1之间，我们使用clamp函数对小于0和大于1的值分别取0和1。
'''
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(PIL_img, image_shape):
    process = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(image_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)
        ]
    )
    return process(PIL_img).unsqueeze(dim=0)  # (batch_size, 3, H, W)

def postprocess(img_tensor):
    inv_normalize = torchvision.transforms.Normalize(
                mean=-rgb_mean/rgb_std,
                std=1/rgb_std
            )
    to_PIL_image = torchvision.transforms.ToPILImage()
    return to_PIL_image(inv_normalize(img_tensor[0].cpu()).clamp(0, 1))


'''
抽取特征:
我们使用基于ImageNet数据集预训练的VGG-19模型来抽取图像特征。
'''
pretrained_net = torchvision.models.vgg19(pretrained=True, progress=True)

'''
为了抽取图像的内容特征和样式特征，我们可以选择VGG网络中某些层的输出。一般来说，越靠近输入层的输出越容易抽取图像的细节信息，
反之则越容易抽取图像的全局信息。为了避免合成图像过多保留内容图像的细节，我们选择VGG较靠近输出的层，也称内容层，来输出图像的内容特征。
我们还从VGG中选择不同层的输出来匹配局部和全局的样式，这些层也叫样式层。在5.7节（使用重复元素的网络（VGG））中我们曾介绍过，
VGG网络使用了5个卷积块。实验中，我们选择第四卷积块的最后一个卷积层作为内容层，以及每个卷积块的第一个卷积层作为样式层。
这些层的索引可以通过打印pretrained_net实例来获取。
'''
print(pretrained_net)

style_layers, content_layers = [0, 5, 10, 19, 28], [25]

'''
在抽取特征时，我们只需要用到VGG从输入层到最靠近输出层的内容层或样式层之间的所有层。下面构建一个新的网络net，
它只保留需要用到的VGG的所有层。我们将使用net来抽取特征
'''
net_list = []
for i in range(max(content_layers + style_layers) + 1):
    net_list.append(pretrained_net.features[i])
net = torch.nn.Sequential(*net_list)

print(net)

'''
给定输入X，如果简单调用前向计算net(X)，只能获得最后一层的输出。由于我们还需要中间层的输出，
因此这里我们逐层计算，并保留内容层和样式层的输出。
'''
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


'''
下面定义两个函数，其中get_contents函数对内容图像抽取内容特征，而get_styles函数则对样式图像抽取样式特征。
因为在训练时无须改变预训练的VGG的模型参数，所以我们可以在训练开始之前就提取出内容图像的内容特征，以及样式图像的样式特征。
由于合成图像是样式迁移所需迭代的模型参数，我们只能在训练过程中通过调用extract_features函数来抽取合成图像的内容特征和样式特征。
'''

def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, style_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, style_Y

'''
定义损失函数：
下面我们来描述样式迁移的损失函数。它由内容损失、样式损失和总变差损失3部分组成。
'''

'''
1.内容损失:
与线性回归中的损失函数类似，内容损失通过平方误差函数衡量合成图像与内容图像在内容特征上的差异。
平方误差函数的两个输入均为extract_features函数计算所得到的内容层的输出。
'''
def content_loss(Y_hat, Y):
    return F.mse_loss(Y_hat, Y)


'''
2.样式损失:
样式损失也一样通过平方误差函数衡量合成图像与样式图像在样式上的差异。为了表达样式层输出的样式，我们先通过extract_features函数计算
样式层的输出。假设该输出的样本数为1，通道数为c，高和宽分别为h和w，我们可以把输出变换成c行hw列的矩阵X。矩阵X可以看作是由c个长度为hw
的向量x1,…,xc组成的。其中向量xi代表了通道i上的样式特征。这些向量的格拉姆矩阵（Gram matrix）XX⊤∈Rc×c中i行j列的元素xij即向量xi与xj
的内积，它表达了通道i和通道j上样式特征的相关性。我们用这样的格拉姆矩阵表达样式层输出的样式。需要注意的是，当hw的值较大时，格拉姆矩阵
中的元素容易出现较大的值。此外，格拉姆矩阵的高和宽皆为通道数c。为了让样式损失不受这些值的大小影响，下面定义的gram函数将格拉姆矩阵除以
了矩阵中元素的个数，即chw。
'''
def gram(X):
    num_channels, n = X.shape[1], X.shape[2] * X.shape[3]   # c, hw
    X = X.view(num_channels, n)
    return torch.matmul(X, X.t()) / (num_channels * n)


'''
自然地，样式损失的平方误差函数的两个格拉姆矩阵输入分别基于合成图像与样式图像的样式层输出。
这里假设基于样式图像的格拉姆矩阵gram_Y已经预先计算好了。
'''
def style_loss(Y_hat, gram_Y):
    return F.mse_loss(gram(Y_hat), gram_Y)

'''
3.总变差损失:
有时候，我们学到的合成图像里面有大量高频噪点，即有特别亮或者特别暗的颗粒像素。一种常用的降噪方法是总变差降噪
（total variation denoising）。假设xi,j表示坐标为(i,j)(i,j)的像素值，降低总变差损失
∑i,j∣xi,j−xi+1,j∣+∣xi,j−xi,j+1∣
能够尽可能使邻近的像素值相似。
'''
def tv_loss(Y_hat):
    return 0.5 * (F.l1_loss(Y_hat[:, :, 1:, :], Y_hat[:, :, :-1, :]) +
                  F.l1_loss(Y_hat[:, :, :, 1:], Y_hat[:, :, :, :-1]))


'''
损失函数:
样式迁移的损失函数即内容损失、样式损失和总变差损失的加权和。通过调节这些权值超参数，我们可以权衡合成图像在保留内容、
迁移样式以及降噪三方面的相对重要性。
'''
content_weight, style_weight, tv_weight = 1, 1e3, 10
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、样式损失和总变差损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(styles_l) + sum(contents_l) + tv_l
    return contents_l, styles_l, tv_l, l


'''
创建和初始化合成图像:
在样式迁移中，合成图像是唯一需要更新的变量。因此，我们可以定义一个简单的模型GeneratedImage，并将合成图像视为模型参数。
模型的前向计算只需返回模型参数即可。
'''
class GeneratedImage(torch.nn.Module):
    def __init__(self, img_shape):
        super(GeneratedImage, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


'''
下面，我们定义get_inits函数。该函数创建了合成图像的模型实例，并将其初始化为图像X。
样式图像在各个样式层的格拉姆矩阵styles_Y_gram将在训练前预先计算好。
'''
def get_inits(X, device, lr, styles_Y):
    gen_img = GeneratedImage(X.shape).to(device)
    gen_img.weight.data = X.data
    optimizer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, optimizer


'''
训练:
在训练模型时，我们不断抽取合成图像的内容特征和样式特征，并计算损失函数。
'''
def train(X, contents_Y, styles_Y, device, lr, max_epochs, lr_decay_epoch):
    print("training on ", device)
    X, styles_Y_gram, optimizer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, gamma=0.1)
    for i in range(max_epochs):
        start = time.time()

        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)

        optimizer.zero_grad()
        l.backward(retain_graph=True)
        # 若在当前backward()后，不执行forward() 而是执行另一个backward()，需要在当前backward()时，
        # 指定保留计算图，backward(retain_graph)
        optimizer.step()
        scheduler.step()

        if i % 50 == 0 and i != 0:
            print('epoch %3d, content loss %.2f, style loss %.2f, '
                  'TV loss %.2f, %.2f sec'
                  % (i, sum(contents_l).item(), sum(styles_l).item(), tv_l.item(),
                     time.time() - start))
    return X.detach()



image_shape =  (150, 225)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
style_X, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.01, 500, 200)

plt.imshow(postprocess(output))

# image_shape = (300, 450)
# _, content_Y = get_contents(image_shape, device)
# _, style_Y = get_styles(image_shape, device)
# X = preprocess(postprocess(output), image_shape).to(device)
# big_output = train(X, content_Y, style_Y, device, 0.01, 500, 200)
#
# d2l.set_figsize((7, 5))
# d2l.plt.imshow(postprocess(big_output))

'''
样式迁移常用的损失函数由3部分组成：内容损失使合成图像与内容图像在内容特征上接近，样式损失令合成图像与样式图像在样式特征上接近，
而总变差损失则有助于减少合成图像中的噪点。
可以通过预训练的卷积神经网络来抽取图像的特征，并通过最小化损失函数来不断更新合成图像。
用格拉姆矩阵表达样式层输出的样式。
'''