from PIL import Image
import numpy as np
import math
import torch

import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l
import matplotlib.pyplot as plt

# print(torch.__version__)  #  1.3.1

''''
生成多个锚框：
假设输入图像高为h，宽为w。我们分别以图像的每个像素为中心生成不同形状的锚框。设大小为s∈(0,1]且宽高比为r>，那么锚框的宽和高将分别为ws√r
和hs/√r。当中心位置给定时，已知宽和高的锚框是确定的。

下面我们分别设定好一组大小s1,…,sn和一组宽高比r1,…,rm。如果以每个像素为中心时使用所有的大小与宽高比的组合，
输入图像将一共得到whnm个锚框。虽然这些锚框可能覆盖了所有的真实边界框，但计算复杂度容易过高。因此，我们通常只对包含s1或r1的大小与宽高比的
组合感兴趣，即(s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1)
也就是说，以相同像素为中心的锚框的数量为n+m−1。对于整个输入图像，我们将一共生成wh(n+m−1)个锚框。
'''

d2l.set_figsize()
img = Image.open('img/catdog.jpg')
w, h = img.size
# print("w = %d, h = %d" % (w, h))  # w = 728, h = 561

def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    """
        # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成(xmin, ymin, xmax, ymax).
        https://zh.d2l.ai/chapter_computer-vision/anchor.html
        Args:
            feature_map: torch tensor, Shape: [N, C, H, W].
            sizes: List of sizes (0~1) of generated MultiBoxPriores.
            ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores.
        Returns:
            anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1
    """
    pairs = []
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])  # 包含s1
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])  # 包含r1

    pairs = np.array(pairs)  # 转换成（s1, r）,(s, r1)
    print(pairs)

    ss1 = pairs[:, 0] * pairs[:, 1]  # size * sqrt(ration) 锚框的宽
    ss2 = pairs[:, 0] / pairs[:, 1]  # size / sqrt(ration) 锚框的高
    # print(ss1)
    # print(ss1.shape)  # (5, )
    # print(ss2)
    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2  # 5*4
    print(base_anchors)
    print(base_anchors.shape)
    h, w = feature_map.shape[-2:]  # 取出h,w
    shifts_x = np.arange(0, w) / w   # 728
    shifts_y = np.arange(0, h) / h   # 561
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    # 生成网格点坐标矩阵。比如x有4个元素，y有3个元素，故生成的矩阵为3行4列的矩阵，形状固定，
    # 矩阵shift_x, shift_y的元素对应x，y本身元素的复制，但x作为shift_x的行向量，y作为shift_y的列向量。
    # print(shift_x, shift_y)
    # print(shift_x.shape, shift_y.shape)  # 561*728
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    # print(shift_x.shape)  # (408408, )
    # print(shift_y.shape)  # (408408, )

    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
    # (408408, 4),(x, y, x, y)两对（x, y）计算左上角和右下角的坐标，与base_anchors.reshape((1, -1, 4))相加也就是计算偏移量。
    # 先遍历x, 再遍历y，组成网格矩阵
    print(shifts.shape)  # (408408, 4)
    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))  # (408408, 1, 4) + (1, 5, 4)
    print(anchors.shape)  # (408408, 5, 4)

    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)  # (1, 2042040, 4)
    


X = torch.Tensor(1, 3, h, w)  # 构造输入数据
Y = MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
# print(Y.shape)   # torch.Size([1, 2042040, 4])

'''
我们看到，返回锚框变量y的形状为（1，锚框个数，4）。将锚框变量y的形状变为（图像高，图像宽，以相同像素为中心的锚框个数，4）后，
我们就可以通过指定像素位置来获取所有以该像素为中心的锚框了。下面的例子里我们访问以（250，250）为中心的第一个锚框。它有4个元素，
分别是锚框左上角的x和y轴坐标和右下角的x和y轴坐标，其中x和y轴的坐标值分别已除以图像的宽和高，因此值域均为0和1之间。
'''
boxes = Y.reshape((h, w, 5, 4))
print(boxes[250, 250, 1, :])  # * torch.tensor([w, h, w, h], dtype=torch.float32)

# shifts_x = np.arange(0, 4) / 4   # 728
# shifts_y = np.arange(0, 3) / 3   # 561
# shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
# print(shift_x)  # (4, 3)
# print(shift_y)   # (3, 4)
# shift_x = shift_x.reshape(-1)  # 12
# shift_y = shift_y.reshape(-1)  # 12
# print(shift_x)
# print(shift_y)
# a = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
# print(a)
# print(a.shape)  # (12, 4)

'''
为了描绘图像中以某个像素为中心的所有锚框，我们先定义show_bboxes函数以便在图像上画出多个边界框。
'''
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().cpu().numpy(), color)
        axes.add_patch(rect)   # 把生成图案绘制到画布上
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'  # 字体为白色
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=6, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


'''
刚刚我们看到，变量boxes中x和y轴的坐标值分别已除以图像的宽和高。在绘图时，我们需要恢复锚框的原始坐标值，
并因此定义了变量bbox_scale。现在，我们可以画出图像中以(250, 250)为中心的所有锚框了。可以看到，大小为0.75
且宽高比为1的锚框较好地覆盖了图像中的狗。
'''
d2l.set_figsize()
fig = d2l.plt.imshow(img)
bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)  # (1, 4)
# show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,  # 广播机制
#             ['s=0.75, r=1', 's=0.75, r=2', 's=0.75, r=0.5', 's=0.5, r=1', 's=0.25, r=1'])

# print(boxes[250, 250, :, :] * bbox_scale)


# a = torch.ones(5, 4) * 2
# b = torch.ones(1, 4) * 3
# print(a * b) # 广播机制

'''
ax.text(x,y,tex_to_write,*kwarg)
其中(x,y)为标注的位置
text_to_write为标注的内容，格式个string，常见使用Latext书写
书写格式为r'$text$'。r表示原始字符串，text按照Latex格式书写即可
如果对Latex公式不是很熟悉，可以查看https://en.wikipedia.org/wiki/Help:Displaying_a_formula
可选参数：
首先时fontdic中的内容，一般设置size或者fontsize=40，style='italic'or'normal'
weight='bold'等。当然可以直接写一个字典：fontdict={'size':30,'weight':'bold'}
继承自text的方法：
（1）alpha可以设置文本透明度范围为0-1
（2）color设置文本的颜色
（3）设置相对位置。水平：ha=‘center’ | ‘right’ | ‘left’
垂直：va= ‘center’ | ‘top’ | ‘bottom’ | ‘baseline’
（4）rotation=45 直接旋转text45度
（5）bbox常见的text包裹用来做提醒
bbox来自FancyBboxPatch，本小节不做过多的介绍
一般使用的属性为facecolor，alpha
'''

'''
交并比：
        ∣A∪B∣
J(A,B)= --------
        ∣A∩B∣
'''
# 以下函数已保存在d2lzh_pytorch包中方便以后使用
# 参考https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py#L356
def compute_intersection(set_1, set_2):
    """
    计算anchor之间的交集
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2) 按元素取最大值
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2) 按元素取最小值
    # print(set_1[:, :2].unsqueeze(1), set_1[:, :2].unsqueeze(1).size())
    # print(set_2[:, :2].unsqueeze(0), set_2[:, :2].unsqueeze(0).size())
    # print(lower_bounds)
    # print('-'*100)
    # print(set_1[:, 2:].unsqueeze(1), set_1[:, 2:].unsqueeze(1).size())
    # print(set_2[:, 2:].unsqueeze(0), set_2[:, 2:].unsqueeze(0).size())
    # print(upper_bounds)
    # print('-'*100)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2) 假如n1=2 n2=2; 最后的交集矩阵为（2，2）
    # print(intersection_dims)
    # print(intersection_dims[:, :, 0])
    # print(intersection_dims[:, :, 1])
    # print('-'*100)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)， 取所有n1,n2维的第0个数据和第一个数据

# a = torch.tensor([[1, 1, 3, 4],
#                   [2, 2, 4, 5]])
# b = torch.tensor([[2, 1, 3, 5],
#                   [1, 3, 4, 4]])
# print(compute_intersection(a, b))

# a = torch.randn((2, 2, 2))
# print(a)
# print(a[:, :, 0:1])
# print(a[:, :, 0])
# print(a[:, :, 1])

def compute_jaccard(set_1, set_2):
    """
    计算anchor之间的Jaccard系数(IoU)
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # Find intersections
    intersection = compute_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


# a = torch.ones(3, 2) * 2
# print(a.unsqueeze(0).size())
# print(a.unsqueeze(1).size())

'''
标注训练集的锚框:
在训练集中，我们将每个锚框视为一个训练样本。为了训练目标检测模型，我们需要为每个锚框标注两类标签：一是锚框所含目标的类别，简称类别；
二是真实边界框相对锚框的偏移量，简称偏移量（offset）。在目标检测时，我们首先生成多个锚框，然后为每个锚框预测类别以及偏移量，
接着根据预测的偏移量调整锚框位置从而得到预测边界框，最后筛选需要输出的预测边界框。
'''

'''
下面演示一个具体的例子。我们为读取的图像中的猫和狗定义真实边界框，其中第一个元素为类别（0为狗，1为猫）,剩余4个元素分别为左上角的x和y轴坐标
以及右下角的xx和yy轴坐标（值域在0到1之间）。这里通过左上角和右下角的坐标构造了5个需要标注的锚框，分别记为A0,…,A4
 （程序中索引从0开始）。先画出这些锚框与真实边界框在图像中的位置。
'''

bbox_scale = torch.tensor((w, h, w, h), dtype=torch.float32)
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],   # 0为狗， 1为猫
                            [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
# show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
# show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])

'''
下面实现MultiBoxTarget函数来为锚框标注类别和偏移量。该函数将背景类别设为0，并令从零开始的目标类别的整数索引自加1（1为狗，2为猫）。
'''
# 以下函数已保存在d2lzh_pytorch包中方便以后使用
def assign_anchor(bb, anchor, jaccard_threshold=0.5):
    """
    # 按照「9.4.1. 生成多个锚框」图9.3所讲为每个anchor分配真实的bb, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        bb: 真实边界框(bounding box), shape:（nb, 4）
        anchor: 待分配的anchor, shape:（na, 4）
        jaccard_threshold: 预先设定的阈值
    Returns:
        assigned_idx: shape: (na, ), 每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
    """
    na = anchor.shape[0]
    nb = bb.shape[0]
    jaccard = compute_jaccard(anchor, bb).detach().cpu().numpy() # shape: (na, nb)
    assigned_idx = np.ones(na) * -1  # 初始全为-1

    # 先为每个bb分配一个anchor(不要求满足jaccard_threshold)
    jaccard_cp = jaccard.copy()
    for j in range(nb):
        i = np.argmax(jaccard_cp[:, j])
        assigned_idx[i] = j
        jaccard_cp[i, :] = float("-inf")  # 赋值为负无穷, 相当于去掉这一行

    # 处理还未被分配的anchor, 要求满足jaccard_threshold
    for i in range(na):
        if assigned_idx[i] == -1:
            j = np.argmax(jaccard[i, :])
            if jaccard[i, j] >= jaccard_threshold:
                assigned_idx[i] = j

    return torch.tensor(assigned_idx, dtype=torch.long)

def xy_to_cxcy(xy):
    """
    将(x_min, y_min, x_max, y_max)形式的anchor转换成(center_x, center_y, w, h)形式的.
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    Args:
        xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    Returns:
        bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h

def MultiBoxTarget(anchor, label):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        anchor: torch tensor, 输入的锚框, 一般是通过MultiBoxPrior生成, shape:（1，锚框总数，4）
        label: 真实标签, shape为(bn, 每张图片最多的真实锚框数, 5)
               第二维中，如果给定图片没有这么多锚框, 可以先用-1填充空白, 最后一维中的元素为[类别标签, 四个坐标值]
    Returns:
        列表, [bbox_offset, bbox_mask, cls_labels]
        bbox_offset: 每个锚框的标注偏移量，形状为(bn，锚框总数*4)
        bbox_mask: 形状同bbox_offset, 每个锚框的掩码, 一一对应上面的偏移量, 负类锚框(背景)对应的掩码均为0, 正类锚框的掩码均为1
        cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)
    """
    assert len(anchor.shape) == 3 and len(label.shape) == 3
    bn = label.shape[0]

    def MultiBoxTarget_one(anc, lab, eps=1e-6):
        """
        MultiBoxTarget函数的辅助函数, 处理batch中的一个
        Args:
            anc: shape of (锚框总数, 4)
            lab: shape of (真实锚框数, 5), 5代表[类别标签, 四个坐标值]
            eps: 一个极小值, 防止log0
        Returns:
            offset: (锚框总数*4, )
            bbox_mask: (锚框总数*4, ), 0代表背景, 1代表非背景
            cls_labels: (锚框总数, 4), 0代表背景
        """
        an = anc.shape[0]
        assigned_idx = assign_anchor(lab[:, 1:], anc)  # (锚框总数, )
        bbox_mask = ((assigned_idx >= 0).float().unsqueeze(-1)).repeat(1, 4)  # (锚框总数, 4)

        cls_labels = torch.zeros(an, dtype=torch.long)  # 0表示背景
        assigned_bb = torch.zeros((an, 4), dtype=torch.float32)  # 所有anchor对应的bb坐标
        for i in range(an):
            bb_idx = assigned_idx[i]
            if bb_idx >= 0:  # 即非背景
                cls_labels[i] = lab[bb_idx, 0].long().item() + 1  # 注意要加一
                assigned_bb[i, :] = lab[bb_idx, 1:]

        center_anc = xy_to_cxcy(anc)  # (center_x, center_y, w, h)
        center_assigned_bb = xy_to_cxcy(assigned_bb)

        offset_xy = 10.0 * (center_assigned_bb[:, :2] - center_anc[:, :2]) / center_anc[:, 2:]  # 计算偏移量
        offset_wh = 5.0 * torch.log(eps + center_assigned_bb[:, 2:] / center_anc[:, 2:])        # 计算偏移量
        offset = torch.cat([offset_xy, offset_wh], dim = 1) * bbox_mask  # (锚框总数, 4)

        return offset.view(-1), bbox_mask.view(-1), cls_labels

    batch_offset = []
    batch_mask = []
    batch_cls_labels = []
    for b in range(bn):
        offset, bbox_mask, cls_labels = MultiBoxTarget_one(anchor[0, :, :], label[b, :, :])

        batch_offset.append(offset)
        batch_mask.append(bbox_mask)
        batch_cls_labels.append(cls_labels)

    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    cls_labels = torch.stack(batch_cls_labels)

    return [bbox_offset, bbox_mask, cls_labels]


'''
我们通过unsqueeze函数为锚框和真实边界框添加样本维。
'''
labels = MultiBoxTarget(anchors.unsqueeze(dim=0),
                        ground_truth.unsqueeze(dim=0))

'''
返回的结果里有3项，均为Tensor。第三项表示为锚框标注的类别。
'''
# print(labels[2])  # tensor([[0, 1, 2, 0, 2]])

'''
我们根据锚框与真实边界框在图像中的位置来分析这些标注的类别。首先，在所有的“锚框—真实边界框”的配对中，锚框A4
与猫的真实边界框的交并比最大，因此锚框A4的类别标注为猫。不考虑锚框A4或猫的真实边界框，在剩余的“锚框—真实边界框”的配对中，
最大交并比的配对为锚框A1和狗的真实边界框，因此锚框A1的类别标注为狗。接下来遍历未标注的剩余3个锚框：与锚框A0
交并比最大的真实边界框的类别为狗，但交并比小于阈值（默认为0.5），因此类别标注为背景；与锚框A2交并比最大的真实边界框的类别为猫，
且交并比大于阈值，因此类别标注为猫；与锚框A3交并比最大的真实边界框的类别为猫，但交并比小于阈值，因此类别标注为背景。
返回值的第二项为掩码（mask）变量，形状为(批量大小, 锚框个数的四倍)。掩码变量中的元素与每个锚框的4个偏移量一一对应。 
由于我们不关心对背景的检测，有关负类的偏移量不应影响目标函数。通过按元素乘法，掩码变量中的0可以在计算目标函数之前过滤掉负类的偏移量。
'''
# print(labels[1])
# tensor([[0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1.,
#          1., 1.]])

'''
返回的第一项是为每个锚框标注的四个偏移量，其中负类锚框的偏移量标注为0。
'''
# print(labels[0])
# tensor([[-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,  1.4000e+00,
#           1.0000e+01,  2.5940e+00,  7.1754e+00, -1.2000e+00,  2.6882e-01,
#           1.6824e+00, -1.5655e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
#          -0.0000e+00, -5.7143e-01, -1.0000e+00,  4.1723e-06,  6.2582e-01]])

'''
输出预测边界框:
在模型预测阶段，我们先为图像生成多个锚框，并为这些锚框一一预测类别和偏移量。随后，我们根据锚框及其预测偏移量得到预测边界框。
当锚框数量较多时，同一个目标上可能会输出较多相似的预测边界框。为了使结果更加简洁，我们可以移除相似的预测边界框。
常用的方法叫作非极大值抑制（non-maximum suppression，NMS）。
'''
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])

offset_preds = torch.tensor([0.0] * (4 * len(anchors)))
cls_probs = torch.tensor([[0., 0., 0., 0.,],  # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
'''
下面我们实现MultiBoxDetection函数来执行非极大值抑制。
'''
# 以下函数已保存在d2lzh_pytorch包中方便以后使用
from collections import namedtuple
Pred_BB_Info = namedtuple("Pred_BB_Info", ["index", "class_id", "confidence", "xyxy"])

def non_max_suppression(bb_info_list, nms_threshold = 0.5):
    """
    非极大抑制处理预测的边界框
    Args:
        bb_info_list: Pred_BB_Info的列表, 包含预测类别、置信度等信息
        nms_threshold: 阈值
    Returns:
        output: Pred_BB_Info的列表, 只保留过滤后的边界框信息
    """
    output = []
    # 先根据置信度从高到低排序
    sorted_bb_info_list = sorted(bb_info_list, key = lambda x: x.confidence, reverse=True)

    while len(sorted_bb_info_list) != 0:
        best = sorted_bb_info_list.pop(0)
        output.append(best)

        if len(sorted_bb_info_list) == 0:
            break

        bb_xyxy = []
        for bb in sorted_bb_info_list:
            bb_xyxy.append(bb.xyxy)

        iou = compute_jaccard(torch.tensor([best.xyxy]),
                              torch.tensor(bb_xyxy))[0] # shape: (len(sorted_bb_info_list), )

        n = len(sorted_bb_info_list)
        sorted_bb_info_list = [sorted_bb_info_list[i] for i in range(n) if iou[i] <= nms_threshold]
    return output

def MultiBoxDetection(cls_prob, loc_pred, anchor, nms_threshold = 0.5):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        cls_prob: 经过softmax后得到的各个锚框的预测概率, shape:(bn, 预测总类别数+1, 锚框个数)
        loc_pred: 预测的各个锚框的偏移量, shape:(bn, 锚框个数*4)
        anchor: MultiBoxPrior输出的默认锚框, shape: (1, 锚框个数, 4)
        nms_threshold: 非极大抑制中的阈值
    Returns:
        所有锚框的信息, shape: (bn, 锚框个数, 6)
        每个锚框信息由[class_id, confidence, xmin, ymin, xmax, ymax]表示
        class_id=-1 表示背景或在非极大值抑制中被移除了
    """
    assert len(cls_prob.shape) == 3 and len(loc_pred.shape) == 2 and len(anchor.shape) == 3
    bn = cls_prob.shape[0]

    def MultiBoxDetection_one(c_p, l_p, anc, nms_threshold = 0.5):
        """
        MultiBoxDetection的辅助函数, 处理batch中的一个
        Args:
            c_p: (预测总类别数+1, 锚框个数)
            l_p: (锚框个数*4, )
            anc: (锚框个数, 4)
            nms_threshold: 非极大抑制中的阈值
        Return:
            output: (锚框个数, 6)
        """
        pred_bb_num = c_p.shape[1]
        anc = (anc + l_p.view(pred_bb_num, 4)).detach().cpu().numpy()  # 加上偏移量

        confidence, class_id = torch.max(c_p, 0)   # 取出最大值，并取出索引号
        confidence = confidence.detach().cpu().numpy()
        class_id = class_id.detach().cpu().numpy()

        pred_bb_info = [Pred_BB_Info(
                            index=i,
                            class_id=class_id[i] - 1,  # 正类label从0开始
                            confidence=confidence[i],
                            xyxy=[*anc[i]])  # xyxy是个列表
                        for i in range(pred_bb_num)]

        # 正类的index
        obj_bb_idx = [bb.index for bb in non_max_suppression(pred_bb_info, nms_threshold)]

        output = []
        for bb in pred_bb_info:
            output.append([
                (bb.class_id if bb.index in obj_bb_idx else -1.0),
                bb.confidence,
                *bb.xyxy
            ])

        return torch.tensor(output)  # shape: (锚框个数, 6)

    batch_output = []
    for b in range(bn):
        batch_output.append(MultiBoxDetection_one(cls_prob[b], loc_pred[b], anchor[0], nms_threshold))

    return torch.stack(batch_output)


'''
然后我们运行MultiBoxDetection函数并设阈值为0.5。这里为输入都增加了样本维。我们看到，返回的结果的形状为(批量大小, 锚框个数, 6)。
其中每一行的6个元素代表同一个预测边界框的输出信息。第一个元素是索引从0开始计数的预测类别（0为狗，1为猫），
其中-1表示背景或在非极大值抑制中被移除。第二个元素是预测边界框的置信度。剩余的4个元素分别是预测边界框左上角的x和y轴坐标以及
右下角的x和y轴坐标（值域在0到1之间）。
'''
output = MultiBoxDetection(
    cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(dim=0),
    anchors.unsqueeze(dim=0), nms_threshold=0.5)

print(output)

'''
我们移除掉类别为-1的预测边界框，并可视化非极大值抑制保留的结果。
'''
fig = d2l.plt.imshow(img)
for i in output[0].detach().cpu().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])   # int(i[0])-> 0->'dog=' , int(i[0])-> 1->'cat='
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)


'''
多尺度目标检测:
在9.4节（锚框）中，我们在实验中以输入图像的每个像素为中心生成多个锚框。这些锚框是对输入图像不同区域的采样。然而，如果以图像每个像素为
中心都生成锚框，很容易生成过多锚框而造成计算量过大。举个例子，假设输入图像的高和宽分别为561像素和728像素，如果以每个像素为中心生成5个
不同形状的锚框，那么一张图像上则需要标注并预测200多万个锚框（561×728×5）。

减少锚框个数并不难。一种简单的方法是在输入图像中均匀采样一小部分像素，并以采样的像素为中心生成锚框。此外，在不同尺度下，我们可以生成
不同数量和不同大小的锚框。值得注意的是，较小目标比较大目标在图像上出现位置的可能性更多。举个简单的例子：形状为1×1、1×2和2×2的目标在
形状为2×2的图像上可能出现的位置分别有4、2和1种。因此，当使用较小锚框来检测较小目标时，我们可以采样较多的区域；而当使用较大锚框来检测
较大目标时，我们可以采样较少的区域。

为了演示如何多尺度生成锚框，我们先读取一张图像。它的高和宽分别为561像素和728像素。
'''
img = Image.open('../../doimg/catdog.jpg')
w, h = img.size # (728, 561)

d2l.set_figsize()

'''
下面定义display_anchors函数。我们在特征图fmap上以每个单元（像素）为中心生成锚框anchors。由于锚框anchors中x和y轴的坐标值分别已除以
特征图fmap的宽和高，这些值域在0和1之间的值表达了锚框在特征图中的相对位置。由于锚框anchors的中心遍布特征图fmap上的所有单元，anchors
的中心在任一图像的空间相对位置一定是均匀分布的。具体来说，当特征图的宽和高分别设为fmap_w和fmap_h时，该函数将在任一图像上均匀采样
fmap_h行fmap_w列个像素，并分别以它们为中心生成大小为s（假设列表s长度为1）的不同宽高比（ratios）的锚框。
'''
def display_anchors(fmap_w, fmap_h, s):
    # 前两维的取值不影响输出结果(原书这里是(1, 10, fmap_w, fmap_h), 我认为错了)
    fmap = torch.zeros((1, 10, fmap_h, fmap_w), dtype=torch.float32)

    # 平移所有锚框使均匀分布在图片上
    offset_x, offset_y = 1.0/fmap_w, 1.0/fmap_h
    anchors = MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5]) + \
        torch.tensor([offset_x/2, offset_y/2, offset_x/2, offset_y/2])

    # print(anchors)
    # print(anchors.size())  # torch.Size([1, 24, 4])
    # print(anchors[0])

    bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
    show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)

# display_anchors(fmap_w=4, fmap_h=2, s=[0.15])
