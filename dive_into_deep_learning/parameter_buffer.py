# -*- coding: utf-8 -*-
# @Time    : 2023/10/20 16:13
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : parameter_buffer.py
# @Software: PyCharm 
# @Comment : register_parameter()和register_buffer()

import torch
import torch.nn as nn

# register_parameter()
class MyModule1(nn.Module):
    def __init__(self):
        super(MyModule1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False)

        self.register_parameter('weight', torch.nn.Parameter(torch.ones(10, 10)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(10)))


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x * self.weight + self.bias
        return x


net = MyModule1()

for name, param in net.named_parameters():
    print(name, param.shape)

print('\n', '*'*40, '\n')

for key, val in net.state_dict().items():
    print(key, val.shape)

print('\n', '='*40, '\n')

# register_buffer()
class MyModule2(nn.Module):
    def __init__(self):
        super(MyModule2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False)

        self.register_buffer('weight', torch.ones(10, 10))
        self.register_buffer('bias', torch.zeros(10))


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x * self.weight + self.bias
        return x


net = MyModule2()

for name, param in net.named_parameters():
    print(name, param.shape)

print('\n', '*'*40, '\n')

for key, val in net.state_dict().items():
    print(key, val.shape)