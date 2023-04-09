# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 16:03
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : pytorch_api.py
# @Software: PyCharm 
# @Comment : Pytorch-API实现线性回归

import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

#1.准备数据
x = torch.rand(500, 1)
y = 0.3*x + 0.8

#2.定义模型
class myLr(nn.Module):
    def __init__(self):
        super(myLr, self).__init__()
        self.Linear = nn.Linear(1, 1)  #nn.Linear(输入的特征数，输出的特征数）

    #自定义模型必须实现forward方法
    def forward(self, x):
        out = self.Linear(x)
        return out

#3.实例化模型，优化器类实例化，loss实例化
my_linear = myLr()  #实例化模型
optimizer = optim.SGD(my_linear.parameters(), lr=1e-3)  #实例化优化器
loss_fn = nn.MSELoss()  #实例化损失函数

#4.循环，进行梯度下降，参数更新
for i in range(3000):
    #得到预测值
    y_predict = my_linear(x)
    loss = loss_fn(y_predict, y)  #注意第一个参数为预测值，第二个参数为真实值

    #梯度置为0
    optimizer.zero_grad()
    #反向传播，计算梯度
    loss.backward()
    #参数更新
    optimizer.step()

    if i%100 == 0:
        params = list(my_linear.parameters())  #取出模型中的参数，进行显示
        print('params:',params)
        print(loss.item(), params[0].item(), params[1].item())


my_linear.eval()
predict = my_linear(x)
plt.scatter(x.data.numpy(), y.data.numpy(), c='r')
plt.plot(x.data.numpy(), predict.data.numpy())
plt.show()