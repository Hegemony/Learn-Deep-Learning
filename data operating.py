import torch
x = torch.empty(5, 3)  # 未初始化的Tensor
print(x)
print('-'*100)

y = torch.empty(3, 2, 3)
print(y)
print('-'*100)

a = torch.zeros(3, 3)  # 全0的Tensor
print(a)
print('-'*100)

b = torch.tensor([5.3, 2])  # 直接根据数据创建tensor
print(b)
print('-'*100)
print('-'*100)
x = torch.rand(5, 3)
y = torch.randn(5, 3)
print(x + y)
print(torch.add(x, y))
print(x)
print(y)
print('-'*100)

print(x)
y = x[0, :]   # 索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。
y += 1
print(y)
print(x)
print(x[0, :])  # 源tensor也被改了
print('-'*100)

x = torch.rand(5, 3)
print(x)
y = x.view(15)  # view()返回的新Tensor与源Tensor虽然可能有不同的size，但是是共享data的，
                # 也即更改其中的一个，另外一个也会跟着改变。
                # (顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)
print(x)
print(y)
x += 1
print(x)
print(y)

x_cp = x.clone().view(15)  # Pytorch还提供了一个reshape()可以改变形状，但是此函数并不能保证返回的是其拷贝，所以不推荐使用。
                           # 推荐先用clone创造一个副本然后再使用view
x -= 1
print(x)
print(x_cp)
print('-'*100)

x = torch.randn(1)
print(x)
print(x.item())  # item(), 它可以将一个标量Tensor转换成一个Python number
print('-'*100)

x = torch.arange(1, 3).view(1, 2)  # 当对两个形状不同的Tensor按元素运算时，可能会触发广播（broadcasting）机制：
                                   # 先适当复制元素使这两个Tensor形状相同后再按元素运算。
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)
print('-'*100)

'''
如果将其属性.requires_grad设置为True，它将开始追踪(track)在其上的所有操作（这样就可以利用链式法则进行梯度传播了）。
完成计算后，可以调用.backward()来完成所有梯度计算。此Tensor的梯度将累积到.grad属性中。

注意在y.backward()时，如果y是标量，则不需要为backward()传入任何参数；否则，需要传入一个与y同形的Tensor

如果不想要被继续追踪，可以调用.detach()将其从追踪记录中分离出来，这样就可以防止将来的计算被追踪，这样梯度就传不过去了。
此外，还可以用with torch.no_grad()将不想被追踪的操作代码块包裹起来，这种方法在评估模型的时候很常用，因为在评估模型时，
我们并不需要计算可训练参数（requires_grad=True）的梯度。
Function是另外一个很重要的类。Tensor和Function互相结合就可以构建一个记录有整个计算过程的有向无环图（DAG）。
每个Tensor都有一个.grad_fn属性，该属性即创建该Tensor的Function, 就是说该Tensor是不是通过某些运算得到的，
若是，则grad_fn返回一个与这些运算相关的对象，否则是None。
'''

x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)
# 该属性即创建该Tensor的Function, 就是说该Tensor是不是通过某些运算得到的，
# 若是，则grad_fn返回一个与这些运算相关的对象，否则是None。
y = x + 2
print(y)
print(y.grad_fn)
# 注意x是直接创建的，所以它没有grad_fn, 而y是通过一个加法操作创建的，所以它有一个为<AddBackward>的grad_fn。
# 像x这种直接创建的称为叶子节点，叶子节点对应的grad_fn是None。
print(x.is_leaf, y.is_leaf)
z = y * y * 3
out = z.mean()
print(z, out)
print('-'*100)

out.backward()
print(x.grad)


print(x)
out2 = x.sum()
print(out2)
out2.backward()
print(x.grad)  # 梯度进行叠加 4.5 + 1 = 5.5
'''
grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，
梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。
'''
out3 = x.sum()
x.grad.data.zero_()  # 梯度清零
out3.backward()
print(x.grad)
print('-'*100)

'''
我们不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量。
所以必要时我们要把张量通过将所有张量的元素加权求和的方式转换为标量，举个例子，假设y由自变量x计算而来，
w是和y同形的张量，则y.backward(w)的含义是：先计算l = torch.sum(y * w)，则l是个标量，然后求l对自变量x的导数
'''

x = torch.rand(5, 3)
y = torch.randn(5, 3)
print(x)
print(y)
print(x * y)