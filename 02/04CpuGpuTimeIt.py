import torch
import timeit

# 准备数据
x = torch.rand(10, 64000)
y = torch.rand(64000, 10240)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_cuda = x.to(device)
y_cuda = y.to(device)
x_cpu = x
y_cpu = y

# 定义要计时的函数
def matmul_cuda():
    torch.mm(x_cuda, y_cuda)

def matmul_cpu():
    torch.mm(x_cpu, y_cpu)

# 使用 timeit 模块进行计时
cuda_time = timeit.timeit("matmul_cuda()", globals=globals(), number=10)
cpu_time = timeit.timeit("matmul_cpu()", globals=globals(), number=10)

print(f"CUDA 计算时间: {cuda_time / 10} 秒")        
print(f"CPU 计算时间: {cpu_time / 10} 秒")

#CUDA 计算时间: 0.002890710400004082 秒
#CPU 计算时间: 0.1175481139999988 秒


