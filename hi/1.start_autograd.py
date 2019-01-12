# 第一节，认识autograd——自动梯度下降
import torch

x = torch.ones(2, 2, requires_grad=True)

y = x+2
print(y.grad_fn)  # 这个grad_fn,就是除了用户创建的张量，其它张量都会有的一个属性，属于Function类

z = y*y*3
out = z.mean()
print(z, '\n', out)  # 一个grad_fn是MulBackward0,一个是MeanBackward1

a = torch.randn(2, 2)
a = ((3*a)/(a-1))
print(a.requires_grad)  # requires_grad属性默认是False，除非你指定它为True
a.requires_grad_(True)  # 作用如其名，表示指定是否需要进行梯度下降
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn, ' ----- ', b.requires_grad)  # grad_fn=SumBackward0 requires_grad=True

out.backward()
print(x.grad)

c = torch.randn(3, requires_grad=True)
d = c*2
while d.data.norm() < 1000:
    d = d*2
print(d)

# gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)  # 不知道啥意思
# d.backward(gradients)
# print(c.grad)

print(c.requires_grad)
print((c**2).requires_grad)
with torch.no_grad():  # 这种用法不会消除requires_grad的True，但可以停止autograd
    print((c**2).requires_grad)
    print(a.requires_grad)
print((c**2).requires_grad)
