import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()
print('-'*100)
'''
访问模型参数
'''
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())

print('-'*100)

for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))

print('-'*100)

'''
初始化模型参数
'''
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)  # 正态分布初始化
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)  # 常数初始化
        print(name, param.data)
print('-'*100)

a = torch.Tensor(2, 3).uniform_(-1, 1)
print(a)
print('-'*100)
'''
自定义权重初始化方法
'''
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

print('-'*100)
'''
共享模型参数
'''
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)

print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))

x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad, net[1].weight.grad)  # 由于没有进行梯度清0，单次梯度是3，两次所以就是6,
print(net[0].weight.data, net[1].weight.data)
print('-'*100)

'''
自定义层
'''

class CenteredLayer(nn.Module):
    # 定义一个不含模型参数的自定义层
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()
print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=float)))

# 也可以使用该层构造更复杂的模型
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
y = net(torch.rand(4, 8))
print(y.mean().item())
print('-'*100)


# 含模型参数的自定义层
'''
ParameterList接收一个Parameter实例的列表作为输入然后得到一个参数列表，使用的时候可以用索引来访问某个参数，
另外也可以使用append和extend在列表后面新增参数。
'''
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x
net = MyDense()
print(net)

'''
arameterDict接收一个Parameter实例的字典作为输入然后得到一个参数字典，然后可以按照字典的规则使用了。
例如使用update()新增参数，使用keys()返回所有键值，使用items()返回所有键值对等等
'''
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)

x = torch.ones(1, 4)
print(net(x, 'linear1'))
print(net(x, 'linear2'))
print(net(x, 'linear3'))