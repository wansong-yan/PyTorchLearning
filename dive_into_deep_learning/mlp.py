# -*- coding: utf-8 -*-
# @Time    : 2023/10/18 16:53
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : mlp.py
# @Software: PyCharm 
# @Comment :

import torch
import matplotlib.pyplot as plt

# 创建一些示例数据
x = torch.linspace(-4, 4, 100)
y = torch.relu(x)

# 分割 x 和 y 的梯度计算
x_detach = x.detach()
y_detach = y.detach()

# 创建绘图
plt.plot(x_detach, y_detach, 'x', label='relu(x)')
plt.legend()  # 显示图例
plt.title('ReLU Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
fig = plt.figure(figsize=(5, 2.5))
plt.show()