import torch
import numpy as np
print(torch.__version__)

#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))

#torch.tensor的广播机制
x_1 = torch.tensor([[1,2]])
x_2 = torch.tensor([[1,2],[3,4]])
y = torch.tensor([1,2])
z_1 = x_1 + y
z_2 = x_2 + y
print("x_1=",x_1)
print("y  =",y)
print("z_1=",z_1)
print("z_2=",z_2)

from math import pi
x_general = torch.tensor([True, 1, pi], dtype=torch.float64)
print(x_general)
print(x_general.dtype)

a = torch.ones((3,4))
print("a = ",a)
b = torch.zeros((3,4))
print("b = ",b)
c = torch.rand((3,4)) 
print("c = ",c)
d = torch.randn((3,4))
print("d = ",d)
e = torch.randint(0,10,(3,4))
print("e = ",e)
print(torch.empty((3,4)))

#torch.tensor和numpy.array的相互转换
x_numpy = np.array([[1,2,3],[4,5,6]])
y_torch = torch.tensor(x_numpy)
x_numpy_2 = np.array(y_torch)
print(type(x_numpy),type(y_torch),type(x_numpy_2))