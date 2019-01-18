import torch
import numpy as np
li = [1, 2, 3, 4, 5]
for i, j in enumerate(li, 2):
    print(i, j)

arr1 = np.array(li)
arr2 = np.array([1, 3, 3, 4, 2])

t = arr1 == arr2
s = t.sum()
sq = t.squeeze()

print(sum(arr1))
print(s)
print(' sq ', sq)


class_total = list(0. for i in range(10))
print(class_total)

a = torch.tensor([0, 0, 1, 0], dtype=torch.uint8)
print(a[2])

for i in ['t', 'v']:
    with torch.set_grad_enabled(i == 't'):
        if i == 't':
            print('aa')
        else:
            print('bb')


