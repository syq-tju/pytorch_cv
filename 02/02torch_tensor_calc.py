import torch

print("*"*40,"张量创建与重整","*"*40)
# 创建一个形状为 [3] 的一维张量
x = torch.tensor([1, 2, 3])
# 增加新的维度使其变为三维张量 [1, 1, 3]
y = x[None, None, :]
# 创建一个形状为 [2, 3] 的二维张量
z = torch.tensor([[10, 20, 30], [40, 50, 60]])
# y 现在是 [1, 1, 3]，可以广播到 [2, 1, 3] 来匹配 z 的 [2, 3]
y_expanded = y.expand(-1, 2, 3)  # 使用 -1 保持那个维度的原始大小
# 尝试广播相加
result = y_expanded + z.unsqueeze(1)
# 输出原始张量和新的张量的形状
print("Tensor y expanded shape:", y_expanded.shape)  # 输出: torch.Size([1, 2, 3])
print("Tensor z shape after unsqueeze:", z.unsqueeze(1).shape)  # 输出: torch.Size([2, 1, 3])
# 输出结果
print("Result of broadcasting addition:\n", result)


print("*"*40,"张量加法","*"*40)
# 张量加法
a = torch.tensor([[1,2,3,4],[5,6,7,8]])
b = 10
c = a + b
d = 2
print("a = ",a)
print("b = ",b)
print("c = ",c)
print(a.add(b))
print("a * 2 = ",a*2)

print(a.shape)
print(a.view(4,2,1))
print(a.view(4,2,1).shape)
print(a)

print("*"*40,"张量乘法","*"*40)
# 张量乘法
e = torch.tensor([[1,2,3],[4,5,6]])
print(e.shape)
f = torch.tensor([[1,2],[3,4],[5,6]])
print(f.shape)
g = torch.matmul(e,f)
print(g)
print(g.shape)
h = torch.mm(e,f)
print(h)
i = e @ f
print(i)

print("*"*40,"张量转置","*"*40)
# 张量转置
j = torch.tensor([[1,2,3],[4,5,6]])
print(j)
print(j.t())
print(j.transpose(0,1))

print("*"*40,"张量拼接","*"*40)
# 张量拼接
k = torch.tensor([[1,2,3],[4,5,6]])
l = torch.tensor([[7,8,9],[10,11,12]])
m = torch.cat((k,l),0)
print(m)
n = torch.cat((k,l),1)
print(n)

print("*"*40,"张量切片","*"*40)
# 张量切片
o = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(o)
print(o[0])
print(o[0,:])
print(o[0:2])
print(o[0:2,0:2])

print("*"*40,"张量索引","*"*40)
# 张量索引
p = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(p)
print(p[0])
print(p[0,:])
print(p[0:2])
print(p[0:2,0:2])

print("*"*40,"求和","*"*40)
# 张量求和
q = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(q)
print(q.sum())
print(q.sum(0))
print(q.sum(1))

print("*"*40,"求平均值","*"*40)
# 张量求平均值
r = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float)
print(r)
print(r.mean())
print(r.mean(0))
print(r.mean(1))

print("*"*40,"求最大值","*"*40)
# 张量求最大值
s = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(s)
print(s.max())
print(s.max(0))
print(s.max(1))

print("*"*40,"求最小值","*"*40)
# 张量求最小值
t = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(t)
print(t.min())
print(t.min(0))
print(t.min(1))

print("*"*40,"求中位数","*"*40)
# 张量求中位数
u = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(u)
print(u.median())
print(u.median(0))
print(u.median(1))

print("*"*40,"求众数","*"*40)
# 张量求众数
v = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(v)
print(v.mode())
print(v.mode(0))
print(v.mode(1))
