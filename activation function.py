import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l

# def xyplot(x_vals, y_vals, name):
#     d2l.set_figsize(figsize=(5, 2.5))
#     d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
#     d2l.plt.xlabel('x')
#     d2l.plt.ylabel(name + '(x)')

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
print(x, y)
print(y.sum())
plt.plot(x.detach().numpy(), y.detach().numpy())

y.sum().backward()
plt.plot(x.detach().numpy(), x.grad.detach().numpy())
plt.show()

