# 第七节，迁移学习
# Author: Sasank Chilamkurthy
# Duplicator: Dy_gs

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def show_pic(tensor, title=None):
    pic = tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pic = std*pic + mean
    pic = np.clip(pic, 0, 1)
    if title is not None:
        plt.title(title)
    plt.imshow(pic)
    plt.pause(0.001)  # 不设置这句pause居然不显示图片了


def train(model, criterion, optimizer, schedule, num_epoch=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0.0

    for epoch in range(num_epoch):
        print('epoch {}/{}'.format(epoch+1, num_epoch))
        print('-'*10)
        for phase in tmp_li:
            if phase == 'train':
                schedule.step()  # schedule是什么
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0.0

            for data, label in data_loader[phase]:
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(data)






if __name__ == '__main__':
    plt.ion()  # 交互模式，开启！
    print('-------start transfer learning--------', time.ctime(), '\n')

    data_tran = {
        'train': transforms.Compose([  # 这里将好几个图片预处理操作组合起来
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),  # 这里和训练数据处理的不一样
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_dir = './data/hymenoptera_data/'
    # 这个作者很喜欢用一行代码代替多行代码，下面连着三行都是这样
    tmp_li = ['train', 'val']
    image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_tran[x]) for x in tmp_li}
    # print(image_dataset)
    data_loader = {x: torch.utils.data.DataLoader(image_dataset[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in tmp_li}
    dataset_size = {x: len(image_dataset[x]) for x in tmp_li}
    class_names = image_dataset[tmp_li[0]].classes

    device = torch.device('cpu')
    # device = torch.device('cuda:0')  # 使用GPU

    # 展示一小批的图片
    train_data, train_label = next(iter(data_loader[tmp_li[0]]))
    out = torchvision.utils.make_grid(train_data)  # 做成几个小格子以便plt.imshow
    show_pic(out, [class_names[x] for x in train_label])

    # 训练！

