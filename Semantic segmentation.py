import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l


'''
Pascal VOC2012语义分割数据集:
语义分割的一个重要数据集叫作Pascal VOC2012 [1]。
其中ImageSets/Segmentation路径包含了指定训练和测试样本的文本文件，而JPEGImages和SegmentationClass路径下分别包含了
样本的输入图像和标签。这里的标签也是图像格式，其尺寸和它所标注的输入图像的尺寸相同。标签中颜色相同的像素属于同一个语义类别。
下面定义read_voc_images函数将输入图像和标签读进内存。
'''
# 本函数已保存在d2lzh_pytorch中方便以后使用
def read_voc_images(root="./Data/VOCdevkit/VOC2012",
                    is_train=True, max_num=None):
    txt_fname = '%s/ImageSets/Segmentation/%s' % (
        root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    if max_num is not None:
        images = images[:min(max_num, len(images))]
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in tqdm(enumerate(images)):
        features[i] = Image.open('%s/JPEGImages/%s.jpg' % (root, fname)).convert("RGB")
        labels[i] = Image.open('%s/SegmentationClass/%s.png' % (root, fname)).convert("RGB")
    return features, labels  # PIL image

voc_dir = "./Data/VOCdevkit/VOC2012"
train_features, train_labels = read_voc_images(voc_dir, max_num=100)

'''
我们画出前5张输入图像和它们的标签。在标签图像中，白色和黑色分别代表边框和背景，而其他不同的颜色则对应不同的类别。
'''
n = 5
imgs = train_features[0:n] + train_labels[0:n]
# print(train_features[0:n])
# print('-'*100)
# print(train_labels[0:n])
# print('-'*100)
# print(imgs)
d2l.show_images(imgs, 2, n)

'''
接下来，我们列出标签中每个RGB颜色的值及其标注的类别。
'''
# 本函数已保存在d2lzh_pytorch中方便以后使用
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
# 本函数已保存在d2lzh_pytorch中方便以后使用
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# print(len(VOC_CLASSES), len(VOC_CLASSES))  # 21, 21

colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
print(colormap2label)
print(colormap2label.size())  # torch.Size([16777216])
for i, colormap in enumerate(VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

# print(colormap2label[128])  # tensor(4, dtype=torch.uint8)

'''
有了上面定义的两个常量以后，我们可以很容易地查找标签中每个像素的类别索引。
'''
# 本函数已保存在d2lzh_pytorch中方便以后使用
def voc_label_indices(colormap, colormap2label):
    """
    convert colormap (PIL image) to colormap2label (uint8 tensor).
    """
    # print(np.array(colormap.convert("RGB")).shape)  # (281, 500, 3)
    colormap = np.array(colormap.convert("RGB")).astype('int32')
    # print(colormap.shape)  # (281, 500, 3)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    # print(len(idx))  # 281
    # print(idx[128])
    # print(idx[128].size)  # 500
    return colormap2label[idx]

y = voc_label_indices(train_labels[0], colormap2label)
# print(y.shape)  # torch.Size([281, 500])
print(y[105:115, 130:140], VOC_CLASSES[1])

'''
预处理数据:
在之前的章节中，我们通过缩放图像使其符合模型的输入形状。然而在语义分割里，这样做需要将预测的像素类别重新映射回原始尺寸的输入图像。
这样的映射难以做到精确，尤其在不同语义的分割区域。为了避免这个问题，我们将图像裁剪成固定尺寸而不是缩放。具体来说，我们使用图像增广
里的随机裁剪，并对输入图像和标签裁剪相同区域。
'''
# 本函数已保存在d2lzh_pytorch中方便以后使用
def voc_rand_crop(feature, label, height, width):
    """
    Random crop feature (PIL image) and label (PIL image).
    """
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            feature, output_size=(height, width))
    # 这个RandomCrop更常用，差别就在于crop时的中心点坐标是随机的，并不是输入图像的中心点坐标，因此基本上每次crop生成的图像都是有差异的
    # print(i, j, w, h)  # 58 110 300 200

    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    # torchvision.transforms.functional.crop(img, i, j, h, w)
    # 裁剪指定PIL图像。
    # 参数：
    # img（PIL图像）– 要裁剪的图像。
    # i – 最上侧像素的坐标。
    # j – 最左侧像素的坐标。
    # h – 要裁剪出的高度。
    # w – 要裁剪出的宽度。

    return feature, label

imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)


d2l.show_images(imgs[::2] + imgs[1::2], 2, n)  # 图片+图片对应的label


# _, axes = plt.subplots(1, 1)
# axes.imshow(imgs[0])
# axes.get_xaxis().set_visible(False)  # 不显示坐标轴
# axes.get_yaxis().set_visible(False)
# plt.show()

# 理解fig, ax = plt.subplots()
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
'''
格式b = a[i:j:s]
这里的s表示步进，缺省为1.（-1时即翻转读取）
所以a[i:j:1]相当于a[i:j]
当s<0时，i缺省时，默认为-1. j缺省时，默认为-len(a)-1
所以a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍。所以你看到一个倒序的东东。
'''
# a = [1, 2, 3, 4, 5, 6, 7]
# print(a[::2])   # [1, 3, 5, 7]
# print(a[1::2])  # [2, 4, 6]

'''
自定义语义分割数据集类:
我们通过继承PyTorch提供的Dataset类自定义了一个语义分割数据集类VOCSegDataset。通过实现__getitem__函数，
我们可以任意访问数据集中索引为idx的输入图像及其每个像素的类别索引。由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输出尺寸，
这些样本需要通过自定义的filter函数所移除。此外，我们还对输入图像的RGB三个通道的值分别做标准化。
'''
# 本函数已保存在d2lzh_pytorch中方便以后使用
class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label, max_num=None):
        """
        crop_size: (h, w)
        """
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean,
                                             std=self.rgb_std)
        ])

        self.crop_size = crop_size  # (h, w)
        features, labels = read_voc_images(root=voc_dir,
                                           is_train=is_train,
                                           max_num=max_num)
        self.features = self.filter(features)  # PIL image
        self.labels = self.filter(labels)      # PIL image
        self.colormap2label = colormap2label
        print('read ' + str(len(self.features)) + ' valid examples')

    def filter(self, imgs):
        return [img for img in imgs if (
            img.size[1] >= self.crop_size[0] and
            img.size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)

        return (self.tsf(feature),  # float32 tensor
                voc_label_indices(label, self.colormap2label))  # uint8 tensor

    def __len__(self):
        return len(self.features)


'''
读取数据集:
我们通过自定义的VOCSegDataset类来分别创建训练集和测试集的实例。假设我们指定随机裁剪的输出图像的形状为320×480。
下面我们可以查看训练集和测试集所保留的样本个数。
'''
crop_size = (320, 480)
max_num = 100
voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label, max_num)
voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label, max_num)

'''
设批量大小为64，分别定义训练集和测试集的迭代器。
'''
batch_size = 64
num_workers = 0
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(voc_test, batch_size, drop_last=True,
                             num_workers=num_workers)

'''
打印第一个小批量的类型和形状。不同于图像分类和目标识别，这里的标签是一个三维数组。
'''
for X, Y in train_iter:
    print(X.dtype, X.shape)
    print(y.dtype, Y.shape)
    break

'''
语义分割关注如何将图像分割成属于不同语义类别的区域。
语义分割的一个重要数据集叫作Pascal VOC2012。
由于语义分割的输入图像和标签在像素上一一对应，所以将图像随机裁剪成固定尺寸而不是缩放。
'''
