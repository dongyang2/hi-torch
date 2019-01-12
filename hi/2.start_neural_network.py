# 第二节，写个网络试试
import torch
import torch.nn as nn
import torch.nn.functional as fun


class Net(nn.Module):
    # 网络结构
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 前向传播
    '''You just have to define the forward function, and the backward function (where gradients are computed) is
    automatically defined for you using autograd. You can use any of the Tensor operations in the forward function.'''
    def forward(self, x):
        x = fun.max_pool2d(fun.relu(self.conv1(x)), (2, 2))  # 把x传入，在第一个卷积层后面加一个pool层
        x = fun.max_pool2d(fun.relu(self.conv2(x)), 2)  # 传x，第二个卷积层后加pool层
        x = x.view(-1, self.num_flat_features(x))  # 利用view函数进行形状改变，这里应该是拉平了
        x = fun.relu(self.fc1(x))  # 传x，在第一个全连接层后面加一个relu
        x = fun.relu(self.fc2(x))
        x = self.fc3(x)  # 输出
        return x

    def num_flat_features(self, x):
        """计算特征数量"""
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(net)

params = list(net.parameters())
# print('number of layers ', len(params), '  conv1 .weight is ', params[0].size())

inp = torch.randn(1, 1, 32, 32)
out = net(inp)  # 这里可以直接接受输入，好神奇哦
# print(out)

'''
# Zero the gradient buffers of all parameters and backprops with random gradients
'''
# net.zero_grad()
# out.backward(torch.randn(1, 10))  # backward应该是继承下来的

'''
#  NOTE
# torch.nn only supports mini-batches. The entire torch.nn package only supports inputs that are a mini-batch of
# samples, and not a single sample.
# For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
# If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.
'''

target = torch.randn(10)  # 造一个假的目标值
target = target.view(1, -1)  # 把一维的数组外面加一层括号变成二维的
criterion = nn.MSELoss()
loss = criterion(out, target)
# print(loss)
# print(loss.grad_fn)  # mse
# print(loss.grad_fn.next_functions[0][0])  # linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # relu

#  后向传播
# net.zero_grad()
print('conv1.bias.grad before backward', net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward', net.conv1.bias.grad)

#  更新参数
# 朴素更新
# lr = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * lr)  # sub应该是减法操作，所以这里就是相当于 weight = weight - grad*lr
# 调包更新
import torch.optim as opt
optimizer = opt.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()  # 先清缓存
output = net(inp)
loss2 = criterion(output, target)
loss2.backward()
optimizer.step()
