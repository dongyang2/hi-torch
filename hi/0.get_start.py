# 第零节，开始
import torch

x = torch.empty(5, 3)  # 得到形状是(5,3)的未初始化的张量，但我的输出不是全零
# print('x', x)

y = torch.rand(5, 3)  # 得到形状是(5,3)的充满随机数的张量
# print('y', y)

ca1 = torch.cat((x, y), dim=0)
print('ca1', ca1.size())  # (10,3)

ca2 = torch.cat((x, y), dim=1)
print('ca2', ca2.size())  # (5,6)

j = torch.rand(3)
k = torch.rand(3)
ca3 = torch.cat((j, k), dim=0)  # 横着拼接两个张量
# ca4 = torch.cat((j, k), dim=1)  # Dimension out of range (expected to be in range of [-1, 0], but got 1)
print('ca3', ca3.size())  # (6)

z = torch.zeros(5, 3, dtype=torch.long)  # 得到形状是(5,3)的充满零的张量
# print('z', z)

a = torch.tensor([5.5, 3])
# print(a)

b = x.new_ones(5, 3, dtype=torch.double)
# print(b)

c = torch.randn_like(b, dtype=torch.float)
# print(c)

print('c-size', c.size())

e = torch.rand(5, 3)

# print(e+y)  # 可以对张量直接用加号，结果是每个对应元素的相加

# print(torch.add(e, y))  # 和上面加号效果一样

result = torch.empty(5, 3)
torch.add(e, y, out=result)
# print(result)  # 这里的输出和之前两个一样

d = torch.empty(5, 3)
d.copy_(y)
# print(d)
# print(d.add_(e))  # 和之前三个输出一样
# print(y)

# print(d[:, 1])

f = torch.randn(4, 4)
print(f)
g = f.view(16)  # view函数可以改变tensor变量的形状
h = f.view(-1, 8)
print(f.size(), g.size(), h.size())
print(h)

i = torch.randn(1)
# print(i)
# print(i.item())  # item()方法可以让tensor中唯一的那个数取出来当一个Python类型的数字，如果tensor中包含超过一个数会报错

'''# pytorch允许所有tensor使用numpy()方法转为numpy类型'''
# print(type(h.numpy()), type(h))
'''但是最常见的用法是下面这句'''
print(h.cpu().detach().numpy())

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # 使用to()方法可以指定设备运行tensor
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

m = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(m.narrow(0, 0, 1))  # narrow(dim,start,len) dim决定了是按行选还是按列选，0=行 1=列。
print(m.narrow(0, 0, 2))  # start是开始的角标，len是角标往后推的长度
print(m.narrow(0, 1, 2))
print(m.narrow(1, 0, 2))
print(m.narrow(1, 1, 2))
