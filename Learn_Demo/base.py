# -*- coding: utf-8 -*-
# @Time    : 2023/4/28 19:11
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : base.py
# @Software: PyCharm 
# @Comment : tensor是什么 tensor四则运算 tensor广播

import torch
import numpy as np
#tensor Random initialization
x = torch.rand(4,3) #rand() is Uniform distribution of [0,1). randn() is N(0,1)
print(x)
y = torch.rand(4,3)
print(y)

#Randomly initialized to 0
a = torch.zeros((4,4), dtype = torch.long)
print(a)
#Randomly initialized to 1
b = torch.ones(4,4)
print(b)
c = torch.tensor(np.ones((2,3), dtype='int32'))
print(c)

#tensor basic operations(op)
#add
print(a + b)
#add_ = replace in (op)
y = a.add_(3)
print(y)

#Index operations
x = torch.rand(3,4)
print(x)
#Second column
print(x[:,1])
#Second row
print(x[1,:])

y = x + 1
print(y[1,:])
print(y)

#Dimension transformation
#Common methods for dimensional transformation of tensors
#torch.view() and torch.reshape()
x = torch.randn(4,3)
y = x.view(12)
z = x.view(-1,6)
print(x.size(),y.size(),z.size())
print(x)
print(y)
print(z)
#x tensor size no change
#view() simply changes the angle of view of this tensor
print(x)

#Hope the original tensor and the transformed tensor will not affect each other.
#For the created tensor and the original tensor do not share memory, using the second method, torch.reshape(),
#It is also possible to change the shape of the tensor,
#but this function does not guarantee that it returns its copy value, so it is officially not recommended
a = torch.randn(4,4)
b = a.reshape(2,8)
print(a)
print(b)

#Broadcast mechanism
x = torch.arange(1,4).view(1,3)
print(x)
y = torch.arange(1,5).view(4,1)
print(y)
print(x + y)