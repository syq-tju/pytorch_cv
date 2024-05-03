import torch
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
