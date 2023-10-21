# -*- coding: utf-8 -*-
# @Time    : 2023/10/21 10:30
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : norm.py
# @Software: PyCharm 
# @Comment : BatchNorm、LayerNorm 和 GroupNorm

# 一般 CNN 中，卷积层后面会跟一个 BatchNorm 层，减少梯度消失和爆炸，提高模型的稳定性。
# Transformer block 中会使用到 LayerNorm ， 一般输入尺寸形为 ：（batch_size, token_num, dim），会在最后一个维度做 归一化： nn.LayerNorm(dim)
# batch size 过大或过小都不适合使用 BN，而是使用 GN。
# （1）当 batch size 过大时，BN 会将所有数据归一化到相同的均值和方差。这可能会导致模型在训练时变得非常不稳定，并且很难收敛。
# （2）当 batch size 过小时，BN 可能无法有效地学习数据的统计信息。

import torch
import torch.nn as nn
import numpy as np

#BatchNorm
bn_feature_array = np.array([[[[1, 0], [0, 2]],
                           [[3, 4], [1, 2]],
                           [[-2, 9], [7, 5]],
                           [[2, 3], [4, 2]]],

                          [[[1, 2], [-1, 0]],
                           [[1, 2], [3, 5]],
                           [[4, 7], [-6, 4]],
                           [[1, 4], [1, 5]]]], dtype=np.float32)

bn_feature_tensor = torch.tensor(bn_feature_array.copy(), dtype=torch.float32)
bn_out = nn.BatchNorm2d(num_features=4, eps=1e-5)(bn_feature_tensor)
print(bn_out)

for i in range(bn_feature_array.shape[1]):
    bn_channel = bn_feature_array[:, i, :, :]
    bn_mean = bn_channel.mean()
    bn_var = bn_channel.var()
    print(bn_mean)
    print(bn_var)

    bn_channel = (bn_channel - bn_mean) / np.sqrt(bn_var + 1e-5)
print(bn_feature_array)

print('\n', '='*40, '\n')

# LayerNorm
ln_feature_array = np.array([[[[1, 0], [0, 2]],
                              [[3, 4], [1, 2]],
                              [[2, 3], [4, 2]]],

                             [[[1, 2], [-1, 0]],
                              [[1, 2], [3, 5]],
                              [[1, 4], [1, 5]]]], dtype=np.float32)

ln_feature_array = ln_feature_array.reshape((2, 3, -1)).transpose(0, 2, 1)
ln_feature_tensor = torch.tensor(ln_feature_array.copy(), dtype=torch.float32)

ln_out = nn.LayerNorm(normalized_shape=3)(ln_feature_tensor)
print(ln_out)

b, token_num, dim = ln_feature_array.shape
ln_feature_array = ln_feature_array.reshape((-1, dim))

for i in range(b * token_num):
    ln_channel = ln_feature_array[i, :]
    ln_mean = ln_channel.mean()
    ln_var = ln_channel.var()
    print(ln_mean)
    print(ln_var)

    ln_channel = (ln_channel - ln_mean) / np.sqrt(ln_var + 1e-5)
print(ln_feature_array.reshape(b, token_num, dim))

print('\n', '='*40, '\n')

# GroupNorm
gn_feature_array = np.array([[[[1, 0], [0, 2]],
                              [[3, 4], [1, 2]],
                              [[-2, 9], [7, 5]],
                              [[2, 3], [4, 2]]],

                             [[[1, 2], [-1, 0]],
                              [[1, 2], [3, 5]],
                              [[4, 7], [-6, 4]],
                              [[1, 4], [1, 5]]]], dtype=np.float32)
gn_feature_tensor = torch.tensor(gn_feature_array.copy(), dtype=torch.float32)
gn_out = nn.GroupNorm(num_groups=2, num_channels=4)(gn_feature_tensor)
print(gn_out)

gn_feature_array = gn_feature_array.reshape((2, 2, 2, 2, 2)).reshape((4, 2, 2, 2))

for i in range(gn_feature_array.shape[0]):
    gn_channel = gn_feature_array[i, :, :, :]
    gn_mean = gn_channel.mean()
    gn_var = gn_channel.var()
    print(gn_mean)
    print(gn_var)

    gn_channel = (gn_channel - gn_mean) / np.sqrt(gn_var + 1e-5)
gn_feature_array = gn_feature_array.reshape((2, 2, 2, 2, 2)).reshape((4, 2, 2, 2))
print(gn_feature_array)

