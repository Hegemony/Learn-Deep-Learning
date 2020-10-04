import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('F:/anaconda3/Lib/site-packages')
import d2lzh_pytorch as d2l

'''
生成数据集
y=1.2x-3.4x^2+5.6x^3+bias
'''

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
# 其中噪声项ϵ服从均值为0、标准差为0.01的正态分布。训练数据集和测试数据集的样本数都设为100。
features = torch.randn(n_test + n_train, 1)
# print(features)
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
# print(poly_features, poly_features.size())
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
# y=1.2x-3.4x^2+5.6x^3+bias
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
# 加入一些噪声

# print(features[:2], poly_features[:2], labels[:2])
'''
定义、训练和测试模型
'''
num_epochs = 100
loss = torch.nn.MSELoss()
# print(poly_features.shape[0], poly_features[:n_train, :].shape[-1])

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)
    # 通过Linear文档可知，pytorch已经将参数初始化了，所以我们这里就不手动初始化了

    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for i in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())

    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    # semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
    #          range(1, num_epochs + 1), test_ls, ['train', 'test'])

    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.semilogy(range(1, num_epochs + 1), train_ls)
    plt.semilogy(range(1, num_epochs + 1), test_ls, linestyle=':')
    # semilogx（X,Y）:将x轴数据以对数建立坐标，Y轴不变
    # semilogy（X,Y）:将y轴数据以对数建立坐标，x轴不变
    plt.legend(['train', 'test'])
    # plt.legend()显示图例，知道哪条线对应哪个
    plt.show()

    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)

fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
            labels[:n_train], labels[n_train:])
