# -*- coding: utf-8 -*-
# @Time    : 2023/5/26 15:41
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : resnet50.py
# @Software: PyCharm 
# @Comment : Sequential
# 1、当模型的前向计算为简单串联各个层的计算时， Sequential 类可以通过更加简单的方式定义模型。
# 2、可以接收一个子模块的有序字典(OrderedDict) 或者一系列子模块作为参数来逐一添加 Module 的实例，模型的前向计算就是将这些实例按添加的顺序逐⼀计算
# 3、使用Sequential定义模型的好处在于简单、易读，同时使用Sequential定义的模型不需要再写forward
import os
import torchvision
import numpy as np
from torch.utils.data import dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
import torch
import collections
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

# nn.Sequential里面的模块按照顺序进行排列的，必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。
# nn.Sequential中可以使用OrderedDict来指定每个module的名字。
net = nn.Sequential(nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))
print(net)
print('--'*50)
net2 = nn.Sequential(collections.OrderedDict([
          ('fc1', nn.Linear(784, 256)),
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(256, 10))
        ]))
print(net2)
print('--'*50)
# ModuleList 接收一个子模块（或层，需属于nn.Module类）的列表作为输入，然后也可以类似List那样进行append和extend操作
# nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起。ModuleList中元素的先后顺序并不代表其在网络中的真实位置顺序
net3 = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net3.append((nn.Linear(256, 10)))
print(net3[-1])
print(net3)
print('--'*50)
# ModuleDict可以像常规Python字典一样索引，同样自动将每个 module 的 parameters 添加到网络之中的容器(注册)。
# 同样的它可以使用OrderedDict、dict或者ModuleDict对它进行update，也就是追加。
net4 = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net4['output'] = nn.Linear(256, 10)
print(net4['linear'])
print(net4.output)
print(net4)
print('=='*50)
# 1. nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList和nn.ModuleDict则没有实现内部forward函数。
# 2. nn.Sequential需要严格按照顺序执行，而其它两个模块则可以任意调用。

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(Block, self).__init__()
        out_channels_01, out_channels_02, out_channels_03 = out_channels
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_01, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels_01),
            nn.ReLU(inplace=True)#对从上层网络Conv2d中传递下来的tensor直接进行修改，节省运算内存，不用多存储其他变量
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels_01, out_channels_02, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels_02),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels_02, out_channels_03, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels_03),
        )
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels_03, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels_03)
            )

    def forward(self, x):
        x_shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x= self.conv3(x)
        if self.downsample:
            x_shortcut = self.shortcut(x_shortcut)
        x = x + x_shortcut
        x = self.relu(x)
        return x

class Resnet50(nn.Module):

    def __init__(self):
        super(Resnet50, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        Layers = [3, 4, 6, 3]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        self.conv2 = self._make_layer(64, (64, 64, 256), Layers[0], 1)
        self.conv3 = self._make_layer(256, (128, 128, 512), Layers[1], 2)
        self.conv4 = self._make_layer(512, (256, 256, 1024), Layers[2], 2)
        self.conv5 = self._make_layer(1024, (512, 512, 2048), Layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 1000)
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        blocks_1 = Block(in_channels, out_channels, stride=stride, downsample=True)
        layers.append(blocks_1)
        for i in range(1, blocks):
            layers.append(Block(out_channels[2], out_channels, stride=1, downsample=False))

        return nn.Sequential(*layers)

resnet_50 = Resnet50()
x = torch.rand((10, 3, 224, 224))
for name, layer in resnet_50.named_children():
    if name != "fc":
        x = layer(x)
        print(name, 'output shaoe:', x.shape)
    else:
        x = x.view(x.size(0), -1)
        x = layer(x)
        print(name, 'output shaoe:', x.shape)

resnet50 = Resnet50()
summary(resnet50, ((10, 3, 224, 224)))

#超参数定义
# 批次的大小
batch_size = 16 #可选32、64、128
# 优化器的学习率
lr = 1e-4
#运行epoch
max_epochs = 2
# 方案一：指定GPU的方式
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 指明调用的GPU为0,1号

# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 指明调用的GPU为1号

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

writer = SummaryWriter('./runs')
# 训练&验证
writer = SummaryWriter('./runs')
# Set fixed random number seed
torch.manual_seed(42)
# 定义损失函数和优化器
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
My_model = Resnet50()
My_model = My_model.to(device)
# 交叉熵
criterion = torch.nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(My_model.parameters(), lr=lr)
epoch = max_epochs

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

        # Write the network graph at epoch 0, batch 0
        if epoch == 0 and iter == 0:
            writer.add_graph(My_model, input_to_model=(images, labels)[0], verbose=True)

        # Write an image at every batch 0
        if iter == 0:
            writer.add_image("Example input", images[0], global_step=epoch)

        outputs = My_model(images)
        loss = criterion(outputs, labels)
        train_total_correct += (outputs.argmax(1) == labels).sum().item()
        # backword
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total_num += labels.shape[0]
        train_total_loss += loss.item()

        # Print statistics
        writer.add_scalar("Loss/Minibatches", train_total_loss, train_total_num)

        print("Epoch [{}/{}], Iter [{}/{}], train_loss:{:4f}".format(i + 1, epoch, iter + 1, total_step,
                                                                     loss.item() / labels.shape[0]))

    # Write loss for epoch
    writer.add_scalar("Loss/Epochs", train_total_loss, epoch)

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