import torch

x = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
z = x.pow(2) + y.pow(2)
print(z)

# z is a tensor, and it has a grad_fn attribute
print(z.grad_fn)

s = z.sum()
print(s)
print(s.grad_fn)

s.backward()
print(x.grad) # ds/dx
print(y.grad) # ds/dy
