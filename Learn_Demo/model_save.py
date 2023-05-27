# -*- coding: utf-8 -*-
# @Time    : 2023/5/27 9:34
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : model_save.py
# @Software: PyCharm 
# @Comment : PyTorch模型保存&读取

# PyTorch存储模型主要采用pkl, pt, pth三种格式
# PyTorch模型主要包含两个部分：模型结构和权重
# 模型是继承nn.Module的类
# 权重的数据结构是一个字典（key是层名，value是权重向量）
# 存储分为两种形式：存储整个模型（包括结构和权重）和只存储模型权重（推荐）
import os
import torch
from torchvision import models
import torch.nn as nn
from collections import OrderedDict

model_50 = models.resnet50(pretrained=True)
save_dir = './resnet50.pth'

# 保存整个模型结构+权重
torch.save(model_50, save_dir)
# 保存模型权重
torch.save(model_50.state_dict, save_dir)
# PyTorch中将模型和数据放到GPU上有两种方式——.cuda()和.to(device)
# 如果要使用多卡训练的话，需要对模型使用torch.nn.DataParallel
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 如果是多卡改成类似0,1,2
model_1 = model_50.cuda()  # 单卡
model_2 = torch.nn.DataParallel(model_1).cuda()  # 多卡
#print(model_2)

# 单卡保存+单卡加载
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_152 = models.resnet152(pretrained=True)
model_152.cuda()

save_dir = 'resnet152.pt'

# 保存+读取整个模型
torch.save(model_152, save_dir)
loaded_model = torch.load(save_dir)
loaded_model.cuda()

# 保存+读取模型权重
torch.save(model_152.state_dict(), save_dir)
# 先加载模型结构
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
# 再加载模型权重
loaded_model.load_state_dict(torch.load(save_dir))
loaded_model.cuda()


# 单卡保存+多卡加载
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号
model = models.resnet152(pretrained=True)
model.cuda()

# 保存+读取整个模型
torch.save(model, save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'   #这里替换成希望使用的GPU编号
loaded_model = torch.load(save_dir)
loaded_model = nn.DataParallel(loaded_model).cuda()

# 保存+读取模型权重
torch.save(model.state_dict(), save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'   #这里替换成希望使用的GPU编号
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model.load_state_dict(torch.load(save_dir))
loaded_model = nn.DataParallel(loaded_model).cuda()


# 多卡保存+单卡加载
# 核心问题是：如何去掉权重字典键名中的"module"，以保证模型的统一性

# 对于加载整个模型，直接提取模型的module属性即可：
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'   #这里替换成希望使用的GPU编号

model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()

# 保存+读取整个模型
torch.save(model, save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号
loaded_model = torch.load(save_dir).module


# 对于加载模型权重，有以下几种思路： 保存模型时保存模型的module属性对应的权重
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'   #这里替换成希望使用的GPU编号

save_dir = 'resnet152.pth'   #保存路径
model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()

# 保存权重
torch.save(model.module.state_dict(), save_dir)
# 这样保存下来的模型参数就和单卡保存的模型参数一样了，可以直接加载。也是比较推荐的一种方法。
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model.load_state_dict(torch.load(save_dir))
loaded_model = nn.DataParallel(loaded_model).cuda()
loaded_model.state_dict = loaded_dict

# 遍历字典去除module
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号
loaded_dict = torch.load(save_dir)
new_state_dict = OrderedDict()
for k, v in loaded_dict.items():
    name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
    new_state_dict[name] = v #新字典的key值对应的value一一对应

loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model.state_dict = new_state_dict
loaded_model = loaded_model.cuda()

# 使用replace操作去除module
loaded_model = models.resnet152()
loaded_dict = torch.load(save_dir)
loaded_model.load_state_dict({k.replace('module.', ''): v for k, v in loaded_dict.items()})


# 多卡加载+多卡保存
# 由于是模型保存和加载都使用的是多卡，因此不存在模型层名前缀不同的问题。
# 但多卡状态下存在一个device（使用的GPU）匹配的问题，
# 即保存整个模型时会同时保存所使用的GPU id等信息，
# 读取时若这些信息和当前使用的GPU信息不符则可能会报错或者程序不按预定状态运行。
# 多卡模式下建议使用权重的方式存储和读取模型：
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'   #这里替换成希望使用的GPU编号

model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()

# 保存+读取模型权重，强烈建议！！
torch.save(model.state_dict(), save_dir)
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model.load_state_dict(torch.load(save_dir))
loaded_model = nn.DataParallel(loaded_model).cuda()

# 如果只有保存的整个模型，也可以采用提取权重的方式构建新的模型：
# 读取整个模型
loaded_whole_model = torch.load(save_dir)
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model.state_dict = loaded_whole_model.state_dict
loaded_model = nn.DataParallel(loaded_model).cuda()