# -*- coding: utf-8 -*-
# @Time    : 2023/9/15 16:05
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : linear_conv1d.py
# @Software: PyCharm 
# @Comment : 1*1的卷积核和全连接层有什么异同？

# 假设卷积层覆盖的局部区域delta=0。在这种情况下，证明卷积内核为每组通道独立地实现一个全连接层。
# 实际就是问，1×1的卷积核是否等价于全连接（参见NiN网络结构）。
# 卷积核为1×1且步长为1的卷积层可以代替全连接层这个表述有一定的误导性，
# 表达为上一层是全连接层的全连接层可以转化为卷积核为1×1的卷积层更合适些；
# 而上一层是卷积层的全连接层可以转化为卷积核为h×w的卷积层，h和w分别为上一层卷积层的高和宽。

# 代码验证
import torch
import torch.nn as nn


class MyNet1(nn.Module):
    def __init__(self, linear1, linear2):
        super(MyNet1, self).__init__()
        self.linear1 = linear1
        self.linear2 = linear2

    def forward(self, X):
        return self.linear2(self.linear1(nn.Flatten()(X)))


class MyNet2(nn.Module):
    def __init__(self, linear, conv2d):
        super(MyNet2, self).__init__()
        self.linear = linear
        self.conv2d = conv2d

    def forward(self, X):
        X = self.linear(nn.Flatten()(X))
        X = X.reshape(X.shape[0], -1, 1, 1)
        X = nn.Flatten()(self.conv2d(X))
        return X


linear1 = nn.Linear(15, 10)
linear2 = nn.Linear(10, 5)
conv2d = nn.Conv2d(10, 5, 1)
linear2.weight = nn.Parameter(conv2d.weight.reshape(linear2.weight.shape))
linear2.bias = nn.Parameter(conv2d.bias)
net1 = MyNet1(linear1, linear2)
net2 = MyNet2(linear1, conv2d)
X = torch.randn(2, 3, 5)
# 两个结果实际存在一定的误差，直接print(net1(X) == net2(X))得到的结果不全是True
print(net1(X))
print(net2(X))