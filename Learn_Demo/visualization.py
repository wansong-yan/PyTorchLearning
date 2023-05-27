# -*- coding: utf-8 -*-
# @Time    : 2023/5/24 19:54
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : visualization.py
# @Software: PyCharm 
# @Comment : 可视化
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

class DemoModel(nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 *5, 20)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = DemoModel()
# 方法一：print打印(模型结构可视化)
print(model)

# 方法二:torchinfo查看 模型结构可视化
# 使用torchinfo.summary()就行了，必需的参数分别是model，input_size[batch_size,channel,h,w]
# 提供了模块信息（每一层的类型、输出shape和参数量）、模型整体的参数量、模型大小、一次前向或者反向传播需要的内存大小等
summary(model, (1, 3, 32, 32))

# 方法三：tensorboard查看
# TensorBoard作为一款可视化工具能够满足 输入数据（尤其是图片）、模型结构、参数分布、debug的需求
# TensorBoard可以记录我们指定的数据，包括模型每一层的feature map，权重，以及训练loss等等
# 利用TensorBoard实现训练过程可视化
# 启动tensorboard
# tensorboard --logdir=/path/to/logs/ --port=xxxx
# 其中“path/to/logs/"是指定的保存tensorboard记录结果的文件路径,等价于上面的“./runs"
# port是外部访问TensorBoard的端口号，可以通过访问ip:port访问tensorboard)
writer = SummaryWriter('./runs')
print(model)
writer.add_graph(model, torch.rand(1, 3, 32, 32))

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
from torchvision import datasets

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

#训练&验证
writer = SummaryWriter('./runs')
 # Set fixed random number seed
torch.manual_seed(42)
# 定义损失函数和优化器
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
My_model = DemoModel()
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

    for iter, (images,labels) in enumerate(train_loader):
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