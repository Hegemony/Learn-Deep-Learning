import torch
from torch import nn

'''
我们可以直接使用save函数和load函数分别存储和读取Tensor。save使用Python的pickle实用程序将对象进行序列化，
然后将序列化的对象保存到disk，使用save可以保存各种对象,包括模型、张量和字典等。而load使用pickle unpickle工具
将pickle的对象文件反序列化为内存。
'''
x = torch.ones(3)
torch.save(x, 'x.pt')
x2 = torch.load('x.pt')
print(x2)

y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)

torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)
print('-'*100)
'''
读写模型：
在PyTorch中，Module的可学习参数(即权重和偏差)，模块模型包含在参数中(通过model.parameters()访问)。
state_dict是一个从参数名称隐射到参数Tesnor的字典对象。
'''
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.output(self.act(self.hidden(x)))
        return a

net = MLP()
print(net)
print(net.state_dict())
# 注意，只有具有可学习参数的层(卷积层、线性层等)才有state_dict中的条目。
# 优化器(optim)也有一个state_dict，其中包含关于优化器状态以及所使用的超参数的信息。
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())

'''
保存和加载模型：
PyTorch中保存和加载训练模型有两种常见的方法:
1.仅保存和加载模型参数(state_dict)；
2.保存和加载整个模型。
'''
# 模型保存：
# torch.save(model.state_dict(), PATH) # 推荐的文件后缀名是pt或pth
# 模型加载：
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))

X = torch.randn(2, 3)
Y = net(X)

PATH = "./xy_net.pt"
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
print(net2)
Y2 = net2(X)
print(Y2)
print(Y2 == Y)
print('-'*100)

'''
PyTorch可以指定用来存储和计算的设备，如使用内存的CPU或者使用显存的GPU。默认情况下，PyTorch会将数据创建在内存，然后利用CPU来计算。
用torch.cuda.is_available()查看GPU是否可用:
'''
print(torch.cuda.is_available())
print(torch.cuda.device_count())  # 查看Gpu数量, 输出 1
print(torch.cuda.current_device())   # 查看当前GPU索引号，索引号从0开始, 输出 0
print(torch.cuda.get_device_name(0))  # 根据索引号查看GPU名字, 输出 'GeForce GTX 1050'
print('-'*100)

'''
Tensor的GPU的计算:
使用.cuda()可以将CPU上的Tensor转换（复制）到GPU上。如果有多块GPU，
我们用.cuda(i)来表示第 i 块GPU及相应的显存（i 从0开始）且cuda(0)和cuda()等价
'''
x = torch.tensor([1, 2, 3])
print(x, x.device)  # 我们可以通过Tensor的device属性来查看该Tensor所在的设备。

# 我们可以直接在创建的时候就指定设备。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1, 2, 3], device=device)
# or
x = torch.tensor([1, 2, 3]).to(device)
print(x)
# 如果对在GPU上的数据进行运算，那么结果还是存放在GPU上。
y = x**2
print(y)
print('-'*100)
'''
需要注意的是，存储在不同位置中的数据是不可以直接进行计算的。
即存放在CPU上的数据不可以直接与存放在GPU上的数据进行运算，位于不同GPU上的数据也是不能直接进行计算的。
'''
# z = y + x.cpu()
# 报错
# RuntimeError: expected device cuda:0 but got device cpu

'''
模型的GPU的计算:
同Tensor类似，PyTorch模型也可以通过.cuda转换到GPU上。我们可以通过检查模型的参数的device属性来查看存放模型的设备。
'''
net = nn.Linear(3, 1)
print(net.parameters())
print(list(net.parameters()))
print(list(net.parameters())[0].device)

net.cuda()  # 将模型从CPU转换到GPU上:
print(list(net.parameters())[0].device)
x = torch.rand(2, 3).cuda()  # 同样的，我们需要保证模型输入的Tensor和模型都在同一设备上，否则会报错。
net(x)
print(net)
print(net(x))