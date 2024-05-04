import torch

x = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

#  out 包含 x 和 y 的计算
out = (x.pow(2) + y.pow(2) + x*y).sum()

out.backward()

print(x.grad)  # 输出: tensor([ 2.,  6., 10.])
print(y.grad)  # 输出: tensor([ 8., 10., 12.])
