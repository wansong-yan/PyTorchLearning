# -*- coding: utf-8 -*-
# @Time    : 2023/9/7 17:07
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : autograd.py
# @Software: PyCharm 
# @Comment :
import torch

print('1.自动梯度计算')
x = torch.arange(4.0, requires_grad=True)  # 1.将梯度附加到想要对其计算偏导数的变量
print('x:', x)
print('x.grad:', x.grad) # 没有参与计算, 为None, 张量由用户手动创建, grad_fn返回结果是None
# torch.dot(x, x) 是 PyTorch 中的一个函数，用于计算两个张量（Tensor）的点积（dot product）。
# 这个函数会沿着第一个参数和第二个参数的维度进行逐元素的乘法，然后将结果相加得到一个标量。
y = 2 * torch.dot(x, x)  # 2.记录目标值的计算
print('y:', y)
y.backward()  # 3.执行它的反向传播函数
print(y.grad_fn)
print('x.grad:', x.grad)  # 4.访问得到的梯度
print('x.grad == 4*x:', x.grad == 4 * x)

## 计算另一个函数
x.grad.zero_()
y = x.sum()
print('y:', y)
y.backward()
print(y.grad_fn)
print('x.grad:', x.grad)

# 非标量变量的反向传播
x.grad.zero_()
print('x:', x)
y = x * x
y.sum().backward()
print('x.grad:', x.grad)


def f(a):
    b = a * 2
    print(b.norm())  # b每个元素平方，然后求和
    while b.norm() < 1000:  # 求L2范数：元素平方和的平方根
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


print('2.Python控制流的梯度计算')
a = torch.tensor(2.0)  # 初始化变量
a.requires_grad_(True)  # 1.将梯度赋给想要对其求偏导数的变量
print('a:', a)
d = f(a)  # 2.记录目标函数
print('d:', d)
d.backward()  # 3.执行目标函数的反向传播函数
print('a.grad:', a.grad)  # 4.获取梯度