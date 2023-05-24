# -*- coding: utf-8 -*-
# @Time    : 2023/5/23 21:04
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : model_build.py
# @Software: PyCharm 
# @Comment :

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F


batch_size = 16
lr = 1e-4
max_epochs = 10
# 方案一：指定GPU的方式
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 指明调用的GPU为0,1号

# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # 指明调用的GPU为1号

#数据读取
#cifar10数据集为例给出构建Dataset类的方法
from torchvision import datasets

#"data_transform"可对图像进行一定的转换，如翻转，裁剪，归一化等操作，可自己定义
data_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                    ])

train_cifar_dataset = datasets.CIFAR10('cifar10', train=True, download=False, transform=data_transform)
test_cifar_dataset = datasets.CIFAR10('cifar10', train=False, download=False, transform=data_transform)

#构建好Dataset后，使用DataLoader按批次读入数据
train_loader = torch.utils.data.DataLoader(train_cifar_dataset,
                                           batch_size=batch_size, num_workers=4,
                                           shuffle=True, drop_last=True)
#dataloader一次性创建num_worker个worker
#并用batch_sampler将指定batch分配给指定worker，worker将它负责的batch加载进RAM。
#然后，dataloader从RAM中找本轮迭代要用的batch，如果找到了，就使用。
#如果没找到，就要num_worker个worker继续加载batch到内存，直到dataloader在RAM中找到目标batch。
#一般情况下都是能找到的，因为batch_sampler指定batch时当然优先指定本轮要用的batch。
#drop_last为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了
#如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。
#shuffer=False表示不打乱数据的顺序，然后以batch为单位从头到尾按顺序取用数据
#shuffer=Ture表示在每一次epoch中都打乱所有数据的顺序，然后以batch为单位从头到尾按顺序取用数据
test_loader = torch.utils.data.DataLoader(test_cifar_dataset,
                                          batch_size=batch_size, num_workers=4,
                                          shuffle=False)

train_cifar_dataset.__getitem__(1)[0].size()

#定义模型
#方法一：预训练模型
Resnet50 = torchvision.models.resnet50(pretrained=True)
Resnet50.fc.out_features = 10
print(Resnet50)
'''
#训练&验证
#定义损失函数和优化器
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#损失函数：交叉熵
criterion = torch.nn.CrossEntropyLoss()
#优化器
optimizer = torch.optim.Adam(Resnet50.parameters(), lr=lr)
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

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total_num += labels.shape[0]
        train_total_loss += loss.item()
        print("Epoch [{}/{}], Iter [{}/{}], train_loss:{:4f}".format(i+1,
                                                                     epoch, iter+1, total_step,
                                                                     loss.item()/labels.shape[0]))
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
        test_total_num += labels.shape[0]
        test_total_loss += loss.item()
    print("Epoch [{}/{}], train_loss:{:.4f}, train_acc:{:.4f}%, test_loss:{:.4f}, test_acc:{:.4f}%".format(i + 1,
                                                                 epoch, train_total_loss / train_total_num,
                                                                 train_total_correct / train_total_num * 100,
                                                                 test_total_loss / test_total_num,
                                                                 test_total_correct / test_total_num *100))
    train_all_loss.append(np.round(train_total_loss / train_total_num, 4))
    test_all_loss.append(np.round(test_total_loss / test_total_num, 4))
'''

#方法二：自定义模型
class DemoModel(nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # nn.Linear是PyTorch中的一个类，用于定义全连接层（Linear层）
        # 作用就是对输入进行线性变换，并添加偏置项。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 *5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#训练&验证
#定义损失函数和优化器
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#损失函数：交叉熵
criterion = torch.nn.CrossEntropyLoss()
#优化器
optimizer = torch.optim.Adam(Resnet50.parameters(), lr=lr)
epoch = max_epochs
My_model = DemoModel()
My_model = My_model.to(device)
total_step = len(train_loader)
train_all_loss = []
test_all_loss = []
for i in range(epoch):
    My_model.train()
    train_total_loss = 0
    train_total_num = 0
    train_total_correct = 0

    for iter, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = My_model(images)
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
        My_model.eval()
        test_total_loss = 0
        test_total_correct = 0
        test_total_num = 0
        for iter, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = My_model(images)
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