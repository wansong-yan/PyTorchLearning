# -*- coding: utf-8 -*-
# @Time    : 2023/4/28 20:28
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : dataLoader.py
# @Software: PyCharm 
# @Comment : 数据读取

#导入常用包
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

# 批次的大小
batch_size = 16 #可选32、64、128
# 优化器的学习率
lr = 1e-4
#运行epoch
max_epochs = 100
# 方案一：指定GPU的方式
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 指明调用的GPU为0,1号

# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # 指明调用的GPU为1号

# 数据读取
#cifar10数据集为例给出构建Dataset类的方式
from torchvision import datasets

#“data_transform”可以对图像进行一定的变换，如翻转、裁剪、归一化等操作，可自己定义
data_transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                   ])

# Dataset类主要包含三个函数：
#
# init: 用于向类中传入外部参数，同时定义样本集
# getitem: 用于逐个读取样本集合中的元素，可以进行一定的变换，并将返回训练/验证所需的数据
# len: 用于返回数据集的样本数

train_cifar_dataset = datasets.CIFAR10('cifar10',train=True, download=False,transform=data_transform)
test_cifar_dataset = datasets.CIFAR10('cifar10',train=False, download=False,transform=data_transform)

#查看dataset
print(test_cifar_dataset.__len__)
image_demo = test_cifar_dataset.__getitem__(1)[0]
print(image_demo)
print(image_demo.size())

#查看数据集
import matplotlib.pyplot as plt
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(test_cifar_dataset)
plt.show()
for i in range(10):
    images, labels = dataiter.__next__()
    print(images.size())
    print(str(classes[labels]))
#     images = images.numpy().transpose(1, 2, 0)  # 把channel那一维放到最后
#     plt.title(str(classes[labels]))
#     plt.imshow(images)

#构建好Dataset后，就可以使用DataLoader来按批次读入数据了
train_loader = torch.utils.data.DataLoader(train_cifar_dataset,
                                           batch_size=batch_size, num_workers=4,
                                           shuffle=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(test_cifar_dataset,
                                         batch_size=batch_size, num_workers=4,
                                         shuffle=False)

# batch_size：样本是按“批”读入的，batch_size就是每次读入的样本数
# num_workers：有多少个进程用于读取数据，Windows下该参数设置为0，Linux下常见的为4或者8，根据自己的电脑配置来设置
# shuffle：是否将读入的数据打乱，一般在训练集中设置为True，验证集中设置为False
# drop_last：对于样本最后一部分没有达到批次数的样本，使其不再参与训练

#自定义 Dataset 类
class MyDataset(Dataset):
    def __init__(self, data_dir, info_csv, image_list, transform=None):
        """
        Args:
            data_dir: path to image directory.
            info_csv: path to the csv file containing image indexes
                with corresponding labels.
            image_list: path to the txt file contains image names to training/validation set
            transform: optional transform to be applied on a sample.
        """
        label_info = pd.read_csv(info_csv)
        image_file = open(image_list).readlines()
        self.data_dir = data_dir
        self.image_file = image_file
        self.label_info = label_info
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_file[index].strip('\n')
        raw_label = self.label_info.loc[self.label_info['Image_index'] == image_name]
        label = raw_label.iloc[:,0]
        image_name = os.path.join(self.data_dir, image_name)
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_file)

#自定义 dataset demo
data_dir = ''
info_csv = ''
image_list = ''
my_dataset = MyDataset(data_dir,info_csv,image_list)