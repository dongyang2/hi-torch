# 第六节，感觉像一个之前几节的集合体
import numpy as np
import torch
import random


def show_clamp():
    a = torch.Tensor([-1.7120,  0.1734, -0.0478, -0.0922])
    print('给定边界[-0.5, 0.1]', a.clamp(-0.5, 0.1))
    print('给定边界[-0.05, ]', a.clamp(min=-0.05))
    print('给定边界[ , 0.1]', a.clamp(max=0.1))


class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return inp.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors  # 这个逗号是什么意思
        grad_in = grad_out.clone()
        grad_in[inp < 0] = 0
        return grad_in


class DynamicNet(torch.nn.Module):
    def __init__(self, di_in, hidden, di_out):  # 接收三个必要的维度
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(di_in, hidden)
        self.middle_linear = torch.nn.Linear(hidden, hidden)
        self.output_linear = torch.nn.Linear(hidden, di_out)

    def forward(self, inp):  # 下面注释是复制的，代码是自己敲的
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        hi_relu = self.input_linear.clamp(min=0)
        for _ in range(random.randint(0, 3)):
            hi_relu = self.middle_linear(hi_relu).clamp(min=0)
        pre = self.output_linear(hi_relu)
        return pre


if __name__ == '__main__':

    # 6.1 Tensor (手动梯度下降)
    batch_size, dimension_in, dimension_hidden, dimension_out = 64, 1000, 100, 10

    x = np.random.randn(batch_size, dimension_in)
    y = np.random.randn(batch_size, dimension_out)

    # Randomly initialize weights
    w1 = np.random.randn(dimension_in, dimension_hidden)
    w2 = np.random.randn(dimension_hidden, dimension_out)

    learning_rate = 1e-6  # 10的-6次方
    for t in range(500):  # 下面这一波操作应该是想手动更新神经网络权重，我复制粘贴过来的
        # Forward pass: compute predicted y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
        if (t+1) % 100 == 0:
            print(t+1, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

    dtype = torch.float
    device = torch.device('cpu')  # 使用CPU
    # device = torch.device('cuda:0')  # 使用GPU

    x = torch.randn(batch_size, dimension_in, device=device, dtype=dtype)
    y = torch.randn(batch_size, dimension_out, device=device, dtype=dtype)
    w1 = torch.randn(dimension_in, dimension_hidden, device=device, dtype=dtype)
    w2 = torch.randn(dimension_hidden, dimension_out, device=device, dtype=dtype)

    for t in range(500):
        h = x.mm(w1)  # mm()是把两个矩阵相乘
        h_relu = h.clamp(min=0)  # clamp()用来把区间外的点打到区间边界上，效果详见show_clamp()
        y_pred = h_relu.mm(w2)

        loss = (y_pred - y).pow(2).sum().item()
        if (t+1) % 100 == 0:
            print(t+1, loss)

        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

    # 6.2 Autograd

    # x = torch.randn(batch_size, dimension_in, device=device, dtype=dtype)
    # y = torch.randn(batch_size, dimension_out, device=device, dtype=dtype)
    #
    # # Create random Tensors for weights.
    # # Setting requires_grad=True indicates that we want to compute gradients with
    # # respect to these Tensors during the backward pass.
    # w1 = torch.randn(dimension_in, dimension_hidden, device=device, dtype=dtype, requires_grad=True)
    # w2 = torch.randn(dimension_hidden, dimension_out, device=device, dtype=dtype, requires_grad=True)
    #
    # for t in range(500):  # 复制粘贴
    #     # Forward pass: compute predicted y using operations on Tensors; these
    #     # are exactly the same operations we used to compute the forward pass using
    #     # Tensors, but we do not need to keep references to intermediate values since
    #     # we are not implementing the backward pass by hand.
    #     y_pred = x.mm(w1).clamp(min=0).mm(w2)
    #
    #     # Compute and print loss using operations on Tensors.
    #     # Now loss is a Tensor of shape (1,)
    #     # loss.item() gets the a scalar value held in the loss.
    #     loss = (y_pred - y).pow(2).sum()
    #     if (t+1) % 100 == 0:
    #         print(t+1, loss.item())
    #
    #     # Use autograd to compute the backward pass. This call will compute the
    #     # gradient of loss with respect to all Tensors with requires_grad=True.
    #     # After this call w1.grad and w2.grad will be Tensors holding the gradient
    #     # of the loss with respect to w1 and w2 respectively.
    #     loss.backward()
    #
    #     # Manually update weights using gradient descent. Wrap in torch.no_grad()
    #     # because weights have requires_grad=True, but we don't need to track this
    #     # in autograd.
    #     # An alternative way is to operate on weight.data and weight.grad.data.
    #     # Recall that tensor.data gives a tensor that shares the storage with
    #     # tensor, but doesn't track history.
    #     # You can also use torch.optim.SGD to achieve this.
    #     with torch.no_grad():  # 看来no_grad()不仅可以拿来预测，也能拿来手动更新网络权重
    #         w1 -= learning_rate * w1.grad
    #         w2 -= learning_rate * w2.grad
    #
    #         # Manually zero the gradients after updating weights
    #         w1.grad.zero_()
    #         w2.grad.zero_()

    x = torch.randn(batch_size, dimension_in, device=device, dtype=dtype)
    y = torch.randn(batch_size, dimension_out, device=device, dtype=dtype)

    w1 = torch.randn(dimension_in, dimension_hidden, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(dimension_hidden, dimension_out, device=device, dtype=dtype, requires_grad=True)

    for t in range(500):  # 多了一个reLU
        relu = MyReLU.apply
        y_pred = relu(x.mm(w1)).mm(w2)

        loss = (y_pred - y).pow(2).sum()
        if (t+1) % 100 == 0:
            print(t+1, loss.item())
        loss.backward()
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            w1.grad.zero_()
            w2.grad.zero_()

    # 6.3 Nn Module

    x = torch.randn(batch_size, dimension_in)
    y = torch.randn(batch_size, dimension_out)

    model = torch.nn.Sequential(
        torch.nn.Linear(dimension_in, dimension_hidden),  # 全连接
        torch.nn.ReLU(),
        torch.nn.Linear(dimension_hidden, dimension_out)  # 全连接
    )  # 这里就已经把模型的结构定义好了

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 调包更权1，torch自带很多优化方法

    for t in range(500):
        predict = model(x)
        loss = criterion(predict, y)
        if (t+1) % 100 == 0:
            print(t+1, loss.item())

        # model.zero_grad()  # Zero the gradients before running the backward pass.
        optimizer.zero_grad()  # 调包更权2，这里就不用zero model里的梯度了，因为optimizer里有所有网络权重
        loss.backward()

        # 手动更权
        # # Update the weights using gradient descent. Each parameter is a Tensor, so
        # # we can access its gradients like we did before.
        # with torch.no_grad():
        #     for param in model.parameters():
        #         param -= learning_rate*param.grad

        # 调包更权3，是不是很爽？
        optimizer.step()

    # 然后官方文档上还给出了自定义继承了nn类的自定义类，同第二节大同小异，不写了也不复制粘贴了。直接最后一步。
    # 展示一下pytorch的动态图的权重共享。
    x = torch.randn(batch_size, dimension_in)
    y = torch.randn(batch_size, dimension_out)

    model = DynamicNet(dimension_in, dimension_hidden, dimension_out)
    # 上面有一个criterion,这里就省略初始化了
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    for i in range(500):  # 训练五部曲
        predict = model(x)            # 第一步，得预测值
        loss = criterion(predict, y)  # 第二步，算损失
        if (i+1) % 100 == 0:
            print(i+1, loss)
        optimizer.zero_grad()         # 第三步，清梯度
        loss.backward()               # 第四步，后向传播
        optimizer.step()              # 第五步，更新权重（按照损失函数优化）
