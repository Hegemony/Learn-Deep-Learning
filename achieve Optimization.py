import sys
import matplotlib.pyplot as plt
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l
from mpl_toolkits import mplot3d  # 三维画图
import numpy as np

def f(x):
    return x * np.cos(np.pi*x)
'''
局部最小值
'''
'''
画法一
'''
x = np.arange(-1.0, 2.0, 0.1)

# fig,  = plt.plot(x, f(x))
# # print(fig, plt.plot(x, f(x)))  # Line2D(_line0)  [<matplotlib.lines.Line2D object at 0x00000198A2697A08>]
# plt.xlabel('x')
# plt.ylabel('f(x)')
# fig.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.0),
#                   arrowprops=dict(arrowstyle='->'))
# fig.axes.annotate('global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8),
#                   arrowprops=dict(arrowstyle='->'))
# plt.show()
'''
Axes.annotate(s, xy, *args, **kwargs)
s：注释文本的内容
xy：被注释的坐标点，二维元组形如(x,y)
xytext：注释文本的坐标点，也是二维元组，默认与xy相同
arrowprops：箭头的样式，dict（字典）型数据，如果该属性非空，则会在注释文本和被注释点之间画一个箭头。
'''

'''
画法二：
先ax = plt.figure() 建一个画布
然后再调用add_subplot()确定画图大小返回一个对象，通过该对象进行画图
'''
ax = plt.figure()
fig = ax.add_subplot(211)  # 一行一列第一个子图
fig.plot(x, f(x))
plt.xlabel('x')
plt.ylabel('f(x)')
fig.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.0),
                  arrowprops=dict(arrowstyle='->'))
fig.axes.annotate('global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8),
                  arrowprops=dict(arrowstyle='->'))


# fig1 = plt.figure()
# ax = fig1.add_subplot(111)
# t = np.arange(0.0, 5.0, 0.01)
# s = np.cos(2*np.pi*t)
# line, = ax.plot(t, s, lw=2)
#
# ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
# ax.set_ylim(-2, 2)
# plt.show()


'''
鞍点
'''
x = np.arange(-2.0, 2.0, 0.1)
fig = ax.add_subplot(212)  # 两行一列第二个子图
fig, = plt.plot(x, x**3)
fig.axes.annotate('saddle point', xy=(0, -0.2), xytext=(-0.52, -5.0),
                  arrowprops=dict(arrowstyle='->'))
plt.xlabel('x')
plt.ylabel('f(x)')


x, y = np.mgrid[-1: 1: 40j, -1: 1: 42j]
# print(x)
print(x.shape)  # 40*42
# print(y)
print(y.shape)  # 40*42
z = x**2 - y**2
# print(z)
print(z.shape)   # 40*42

ax = plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})  # 画3d图
# ax.plot_wireframe(点阵X坐标矩阵，点阵Y坐标矩阵，点阵Z坐标矩阵，rstride=行跨距，cstride=列跨距，linewidth=线宽，color=颜色）
# 行跨距和列跨距：线条之间的距离，值越小，越密集，但绘制速度越慢。
ax.plot([0], [0], [0], 'rx')  # 标记鞍点的位置
ticks = [-1,  0, 1]  # 标注坐标的刻度
plt.xticks(ticks)  # x轴
plt.yticks(ticks)  # y轴
ax.set_zticks(ticks)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

'''
np.mgrid的用法:
功能：返回多维结构，常见的如2D图形，3D图形
np.mgrid[ 第1维，第2维 ，第3维 ， …] 
第n维的书写形式为：
a:b:c
c表示步长，为实数表示间隔；该为长度为[a,b),左开右闭
或：
a:b:cj
cj表示步长，为复数表示点数；该长度为[a,b]，左闭右闭
结果值先y向右扩展，再x向下扩展
例:
import numpy as np
x, y = np.mgrid[1:3:3j, 4:5:2j]
x
x返回：
array([[1., 1.],
       [2., 2.],
       [3., 3.]])
输出y:
array([[4., 5.],
       [4., 5.],
       [4., 5.]])
       
所以表示的结果是：
[[(1,4),(1,5)]
 [(2,4),(2,5)]
 [(3,4),(3,5)]
]
'''
