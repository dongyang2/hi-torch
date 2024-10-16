# 第零节，开始
# 认识张量
import torch
import numpy as np

"""0.1 初始化一个张量"""
x = torch.empty(5, 3)  # 得到形状是(5,3)的未初始化的张量，但我的输出不是全零
print(f"x={x}, \nx.shape={x.shape}, x.size()={x.size()}, x.dtype={x.dtype}, x.device={x.device}")

y = torch.rand(5, 3)  # 得到形状是(5,3)的充满随机数的张量
# print('y', y)

z = torch.zeros(5, 3, dtype=torch.long)  # 得到形状是(5,3)的充满零的张量
# print('z', z)

a = torch.tensor([5.5, 3])  # 输入直接给一个list，那么这个list就可以被转为张量
print(f"a={a}")

b = x.new_ones(5, 3, dtype=torch.double)  # 没错，torch里有double
print(f"b={b}")

c = torch.randn_like(b, dtype=torch.float)
# print(c)

print('c-size', c.size())

d = torch.empty(5, 3)
d.copy_(y)

e = torch.ones(5, 3)
print(f"e={e}")

"""0.2 张量操作 - 拼接，张量内元素求和，矩阵相加的4种方式，修改形状，取数，和numpy类型互转，张量转设备，张量截取"""

ca1 = torch.cat((x, y), dim=0)
print('ca1', ca1.size())  # (10,3)

ca2 = torch.cat((x, y), dim=1)
print('ca2', ca2.size())  # (5,6)

j = torch.rand(3)
k = torch.rand(3)
ca3 = torch.cat((j, k), dim=0)  # 横着拼接两个张量
# ca4 = torch.cat((j, k), dim=1)  # Dimension out of range (expected to be in range of [-1, 0], but got 1)
print('ca3', ca3.size())  # (6)


# --×--矩阵相加--×--
print(f"b+e={b+e}")  # 可以对张量直接用加号，结果是每个对应元素的相加
# print(torch.add(e, y))  # 和上面加号效果一样

result = torch.empty(5, 3)
torch.add(e, y, out=result)
# print(result)  # 这里的输出和之前两个一样

print(d.add_(e))  # 和之前三个输出一样


# --×--张量内元素求和--×--
print(f"b.sum()={b.sum()}")


# --×--修改形状--×--
f = torch.randn(3, 4)
print(f"f={f}\nshape={f.size()}")
# view函数可以改变tensor变量的形状
g = f.view(12)  # 这里12 = 3×4 等于就把 张量f 给展平了
print(f"f.view(12)={g}\nshape={g.size()}")
# 如果view接收2个参数，那么用-1占位，这里view()函数可以把 张量修改为 n×第二个数字 的形状
h = f.view(-1, 4)
print(f"f.view(-1, 4)={h}\nshape={h.size()}")
h1 = f.view(-1, 3)
print(f"f.view(-1, 3)={h1}\nshape={h1.size()}")


# --×--取数--×--
i = torch.randn(1)
print(f"i={i}\ni.item()={i.item()}")  # item()方法可以让tensor中唯一的那个数取出来当一个Python类型的数字，如果tensor中包含超过一个数会报错


# --×--转为numpy--×--
'''# pytorch允许所有tensor使用numpy()方法转为numpy类型'''
# print(type(h.numpy()), type(h))
'''但是最常见的用法是下面这句'''
print(h.cpu().detach().numpy())
print(f"torch.from_numpy(np.zeros(2,3)) = \n{torch.from_numpy(np.zeros([2,3],dtype=int))}")  # numpy转tensor


# --×--转设备--×--
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # 使用to()方法可以指定设备运行tensor
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # to方法不仅可以迁移张量到cpu，而且可以同时对dtype进行更改


# --×--截取--×--
# narrow(dim,start,len)
# dim决定了是按行选还是按列选，0=行 1=列。
# start是开始的角标，len是角标往后推的长度
m = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"开始截取--×--\nm={m}")
print(f"m.narrow(0, 0, 1)={m.narrow(0, 0, 1)}")
print(f"m.narrow(0, 0, 2)={m.narrow(0, 0, 2)}")
print(f"m.narrow(0, 1, 2)={m.narrow(0, 1, 2)}")
print(f"m.narrow(1, 0, 2)={m.narrow(1, 0, 2)}")
print(f"m.narrow(1, 1, 2)={m.narrow(1, 1, 2)}")
