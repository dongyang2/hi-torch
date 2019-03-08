# 第零节，开始
import torch

x = torch.empty(5, 3)
# print(x)

y = torch.rand(5, 3)
# print(y)

z = torch.zeros(5, 3, dtype=torch.long)
# print(z)

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
