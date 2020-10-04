import torch
from time import time


a = torch.ones(1000)
b = torch.ones(1000)

start = time()
# print(start)   # 记录开始的时间
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]  # 标量相加,时间比较慢

print(time() - start)

start = time()
d = a + b
print(time() - start)  # 向量相加，时间更快
print('-'*100)

a = torch.ones(3)
b = 10
print(a + b)
