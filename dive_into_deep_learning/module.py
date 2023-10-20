# -*- coding: utf-8 -*-
# @Time    : 2023/10/20 10:30
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : module.py
# @Software: PyCharm 
# @Comment :

import torch
import torch.nn as nn
# from torchsummary import summary

net1 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
net1.append(nn.Linear(64, 10))

net2 = nn.ModuleList([nn.Linear(32, 64), nn.ReLU()])
for name, param in net2.named_parameters():
    print(name, param.size())

net3 = nn.ModuleDict({'Linear1': nn.Linear(32, 64), 'act': nn.ReLU()})
net3['linear2'] = nn.Linear(64, 128)
for name, param in net3.named_parameters():
    print(name, param.size())

print(net1)
print(net2)
print(net3)

x = torch.randn(8, 3, 32)
print("Sequential net:", net1(x).shape)
# print(net2(x).shape)  # 报错，提示缺少forward
# print(net3(x).shape)   # 报错，提示缺少forward

# 为 nn.ModuleList 写 forward 函数
class My_ModuleList(nn.Module):
    def __init__(self):
        super(My_ModuleList, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(32, 64), nn.ReLU()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

ModuleList_net = My_ModuleList()
x = torch.randn(8, 3, 32)
out = ModuleList_net(x)
print("ModuleList net:", out.shape)

# 为 nn.ModuleDict 写 forward 函数
class My_ModuleDict(nn.Module):
    def __init__(self):
        super(My_ModuleDict, self).__init__()
        self.layers = nn.ModuleDict({'linear': nn.Linear(32, 64), 'act': nn.ReLU()})

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

ModuleDict_net = My_ModuleDict()

x = torch.randn(8, 3, 32)
out = ModuleDict_net(x)
print("ModuleDict net:", out.shape)

# 将 nn.ModuleList 转换成 nn.Sequential
module_list = nn.ModuleList([nn.Linear(32, 64), nn.ReLU()])
List2Seq_net = nn.Sequential(*module_list)
x = torch.randn(8, 3, 32)
print("ModuleList to Sequential net:", List2Seq_net(x).shape)

# 将 nn.ModuleDict 转换成 nn.Sequential
module_dict = nn.ModuleDict({'linear': nn.Linear(32, 64), 'act': nn.ReLU()})
Dict2Seq_net = nn.Sequential(*module_dict.values())
x = torch.randn(8, 3, 32)
print("ModuleDict to Sequential net:", Dict2Seq_net(x).shape)

model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1),
                      nn.BatchNorm2d(16),
                      nn.ReLU())

# print(summary(model, (3, 224, 224), 8))