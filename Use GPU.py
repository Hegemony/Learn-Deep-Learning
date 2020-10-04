import torch
from torch import nn

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