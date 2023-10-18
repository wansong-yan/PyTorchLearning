# -*- coding: utf-8 -*-
# @Time    : 2023/9/16 16:58
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : padding.py
# @Software: PyCharm 
# @Comment : 填充（在输入图像的边界填充元素）和步幅（每次滑动元素的数量）

# 卷积神经网络中卷积核的高度和宽度通常为奇数, 例如1、3、5或7。
# 选择奇数的好处是，保持空间维度的同时，我们可以在顶部和底部填充相同数量的行，在左侧和右侧填充相同数量的列。
#
# 此外，使用奇数的核大小和填充大小也提供了书写上的便利。对于任何二维张量X，当满足：
# 1. 卷积核的大小是奇数；
# 2. 所有边的填充行数和列数相同；
# 3. 输出与输入具有相同高度和宽度
# 则可以得出：输出Y[i, j]是通过以输入X[i, j]为中心，与卷积核进行互相关计算得到的。

import torch
from torch import nn


# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)

# 当卷积核的高度和宽度不同时，我们可以填充不同的高度和宽度，使输出和输入具有相同的高度和宽度。
# 若使用高度为5，宽度为3的卷积核，高度和宽度两边的填充分别为2和1。
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

# 在计算互相关时，卷积窗口从输入张量的左上角开始，向下、向右滑动。
# 在前面的例子中，我们默认每次滑动一个元素。
# 但是，有时候为了高效计算或是缩减采样次数，卷积窗口可以跳过中间位置，每次滑动多个元素。

# 将高度和宽度的步幅设置为2，从而将输入的高度和宽度减半
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)