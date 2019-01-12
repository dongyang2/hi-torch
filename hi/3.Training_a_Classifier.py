# 第三节，训练一个卷积网络作为分类器
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as opt
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 定义网络
class CNN(nn.Module):
    # 网络结构
    def __init__(self):
        super(CNN, self).__init__()  # 这不知道为啥把自己super一下
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)  # 这把pool层放到了网络结构里，和第二节的处理不一样
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 85)
        self.fc3 = nn.Linear(85, 10)  # 经过3个全连接层，让输出变成适合十分类的样子

    def forward(self, inp):
        x = self.pool(fun.relu(self.conv1(inp)))
        x = self.pool(fun.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = fun.relu(self.fc1(x))
        x = fun.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 这里一定要用这个main函数，不然会报BrokenPipeError: [Errno 32] Broken pipe。因为只有用了main才能使用多进程
if __name__ == '__main__':

    print(' --------start---------', time.ctime(), '\n')  # GT940M 6分钟 i5-5300U 3分钟

    batch_size = 4

    # 获得数据集
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = CNN()

    # 定义损失函数和优化方法
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.SGD(net.parameters(), lr=0.001, momentum=0.9)  # momentum我暂时还没理解，知道它叫动量且看了网上的公式

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net.to(device)

    # 训练！
    for epoch in range(2):  # 训练的轮数
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):  # 这里的第二个参数表示从零开始计数，比如enumerate(li, 2)，i就是2,3,4,5······
            train_data, train_label = data

            # # 数据也放到显卡上
            # train_data, train_label = train_data.to(device), train_label.to(device)

            optimizer.zero_grad()
            output = net(train_data)
            loss = criterion(output, train_label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('训练完毕')

    # # 打印真实结果
    # data_iter = iter(test_loader)
    # test_data, test_label = data_iter.__next__()
    # print('GroundTruth: ', ' '.join('%5s' % classes[test_label[j]] for j in range(4)))
    #
    # # 打印预测结果
    # outputs = net(test_data)
    # _, predict = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join('%5s' % classes[predict[j]] for j in range(4)))

    # # 看在整个测试集上的预测效果
    # correct = 0
    # total = 0
    # with torch.no_grad():  # no_grad不会消除requires_grad的True，但可以停止autograd.看来是用在预测阶段的
    #     for data in test_loader:
    #         test_data, test_label = data
    #
    #         # 把数据放入显卡
    #         test_data = test_data.to(device)
    #         test_label = test_label.to(device)
    #
    #         output = net(test_data)
    #         _, predict = torch.max(output, 1)
    #         total += test_label.size(0)
    #         correct += (predict == test_label).sum().item()
    #
    # print('Precision ', 100.0*correct/total)

    # 查看在哪些类上的预测效果不好
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:

            # print(data)

            test_data, test_label = data

            # test_data, test_label = test_data.to(device), test_label.to(device)

            # print(test_data.size(), test_label.size())

            output = net(test_data)
            _, predict = torch.max(output, 1)
            # c = (predict == test_label).squeeze()  # 把数组中 为1的维度 去掉。哈？还是不懂
            c = (predict == test_label)
            # print(c)
            # for i in range(4):  # 这里为什么要循环个4次,因为dataLoader函数的batchSize=4,所以应该写成下面这行的样子
            for i in range(batch_size):
                lab = test_label[i]
                class_correct[lab] += c[i].item()
                class_total[lab] += 1
            # break

    for i in range(10):
        print('Precision of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    print('\n ---------end---------', time.ctime())
