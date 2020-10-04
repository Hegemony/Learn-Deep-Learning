from PIL import Image

import matplotlib.pyplot as plt
import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l

# d2l.set_figsize()
img = Image.open('img/catdog.jpg')
# plt.imshow(img)   # 加分号只显示图
'''
边界框:
在目标检测里，我们通常使用边界框（bounding box）来描述目标位置。边界框是一个矩形框，可以由矩形左上角的x和y轴坐标与
右下角的x和y轴坐标确定。我们根据上面的图的坐标信息来定义图中狗和猫的边界框。图中的坐标原点在图像的左上角，
原点往右和往下分别为x轴和y轴的正方向。
'''
# bbox是bounding box的缩写
dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]

'''
我们可以在图中将边界框画出来，以检查其是否准确。画之前，我们定义一个辅助函数bbox_to_rect。它将边界框表示成matplotlib的边界框格式。
'''
def bbox_to_rect(bbox, color):  # 本函数已保存在d2lzh_pytorch中方便以后使用
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

'''
我们将边界框加载在图像上，可以看到目标的主要轮廓基本在框内。
'''
fig = plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
# subplot()是将整个figure均等分割，而axes()则可以在figure上画图。
# .Axes.add_patch把生成图案绘制到画布上。
'''
Rectangle类
class matplotlib.patches.Rectangle(
    xy, width, height, angle=0.0, **kwargs)
参数:
xy: 2元组 矩形左下角xy坐标;
width:矩形的宽度;
height:矩形的高度;
angle: float, 可选，矩形相对于x轴逆时针旋转角度，默认0;
fill: bool, 可选，是否填充矩形;
'''

'''
在目标检测里不仅需要找出图像里面所有感兴趣的目标，而且要知道它们的位置。位置一般由矩形边界框来表示。
'''
