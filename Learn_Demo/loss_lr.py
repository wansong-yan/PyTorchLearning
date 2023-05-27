# -*- coding: utf-8 -*-
# @Time    : 2023/5/27 11:00
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : loss_lr.py
# @Software: PyCharm 
# @Comment : 1.自定义损失函数 2.动态调整学习率

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import time
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision import datasets

# 以函数方式定义 or 以类方式定义
# 虽然以函数定义的方式很简单，但是以类方式定义更加常用
# 在以类方式定义损失函数时，如果看每一个损失函数的继承关系我们就可以发现Loss函数部分继承自_loss, 部分继承自_WeightedLoss
# 而_WeightedLoss继承自_loss， _loss继承自 nn.Module
# 我们可以将其当作神经网络的一层来对待
# 同样地，我们的损失函数类就需要继承自nn.Module类
# DiceLoss 实现Vent医学影像分割模型的损失函数
class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice_loss

# 使用方法
# criterion = DiceLoss()
# loss = criterion(input, targets)

# 自定义实现多分类损失函数 处理多分类
# cross_entropy + L2正则化
class MyLoss(torch.nn.Module):

    def __init__(self, weight_decay=0.01):
        super(MyLoss, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets)
        l2_loss = torch.tensor(0., requires_grad=True).to(inputs.device)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_loss += torch.norm(param)
        loss = ce_loss + self.weight_decay * l2_loss
        return loss


# def adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_rate, lr_decay_epoch):
#     """根据当前epoch调整学习率
#
#     Args:
#         optimizer: 优化器
#         epoch: 当前epoch
#         init_lr: 初始学习率
#         lr_decay_rate: 学习率衰减率
#         lr_decay_epoch: 学习率衰减间隔
#
#     Returns:
#         当前学习率
#     """
#     current_lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = current_lr
#     return current_lr
# 在第i个epoch结束后更新学习率
# current_lr = adjust_learning_rate(optimizer, i, init_lr, lr_decay_rate, lr_decay_epoch)

def adjust_learning_rate(optim, epoch, size=10, gamma=0.1):
    if (epoch + 1) % size == 0:
        pow = (epoch + 1) // size
        lr = adjust_learning_rate * np.power(gamma, pow)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

#超参数定义
# 批次的大小
batch_size = 16 #可选32、64、128
# 优化器的学习率
lr = 1e-4
#运行epoch
max_epochs = 2
# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # 指明调用的GPU为1号
# 数据读取
#cifar10数据集为例给出构建Dataset类的方式

#“data_transform”可以对图像进行一定的变换，如翻转、裁剪、归一化等操作，可自己定义
data_transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                   ])


train_cifar_dataset = datasets.CIFAR10('cifar10',train=True, download=False,transform=data_transform)
test_cifar_dataset = datasets.CIFAR10('cifar10',train=False, download=False,transform=data_transform)

#构建好Dataset后，就可以使用DataLoader来按批次读入数据了
train_loader = torch.utils.data.DataLoader(train_cifar_dataset,
                                           batch_size=batch_size, num_workers=4,
                                           shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(test_cifar_dataset,
                                         batch_size=batch_size, num_workers=4,
                                         shuffle=False)
# restnet50 pretrained
Resnet50 = torchvision.models.resnet50(pretrained=True)
Resnet50.fc.out_features=10
print(Resnet50)

# 训练&验证
writer = SummaryWriter("../train_skills")
# 定义损失函数和优化器
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 损失函数：自定义损失函数
criterion = MyLoss()
# 优化器
optimizer = torch.optim.Adam(Resnet50.parameters(), lr=lr)

#自定义 scheduler
scheduler_my = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1), verbose=True)
print("初始化的学习率: ", optimizer.defaults['lr'])

epoch = max_epochs
Resnet50 = Resnet50.to(device)
total_step = len(train_loader)
train_all_loss = []
test_all_loss = []

for i in range(epoch):
    Resnet50.train()
    train_total_loss = 0
    train_total_num = 0
    train_total_correct = 0

    for iter, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = Resnet50(images)
        loss = criterion(outputs, labels)
        train_total_correct += (outputs.argmax(1) == labels).sum().item()

        # backword
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total_num += labels.shape[0]
        train_total_loss += loss.item()
        print("Epoch [{}/{}], Iter [{}/{}], train_loss:{:4f}".format(i + 1, epoch, iter + 1, total_step,
                                                                     loss.item() / labels.shape[0]))
    writer.add_scalar("lr", optim.param_group[0]['lr'], i)

    print("第%d个epoch的学习率: %f" % (epoch, optimizer.param_groups[0]['lr']))
    scheduler_my.step()
    #自定义调整lr
    adjust_learning_rate(optimizer, i)

    Resnet50.eval()
    test_total_loss = 0
    test_total_correct = 0
    test_total_num = 0
    for iter, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = Resnet50(images)
        loss = criterion(outputs, labels)
        test_total_correct += (outputs.argmax(1) == labels).sum().item()
        test_total_loss += loss.item()
        test_total_num += labels.shape[0]
    print("Epoch [{}/{}], train_loss:{:.4f}, train_acc:{:.4f}%, test_loss:{:.4f}, test_acc:{:.4f}%".format(
        i + 1, epoch, train_total_loss / train_total_num, train_total_correct / train_total_num * 100,
        test_total_loss / test_total_num, test_total_correct / test_total_num * 100

    ))
    train_all_loss.append(np.round(train_total_loss / train_total_num, 4))
    test_all_loss.append(np.round(test_total_loss / test_total_num, 4))