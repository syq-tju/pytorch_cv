import torch                                            # 导入 PyTorch 库
import torch.nn as nn                                   # 导入神经网络模块
from torch.utils.data import DataLoader, Dataset        # 导入数据集和数据加载器

x = [[1,2],[3,4],[5,6],[7,8]]                           # 输入数据
y = [[3],[7],[11],[15]]                                 # 输出数据


X = torch.tensor(x, dtype=torch.float32)                # 将数据转换为张量
Y = torch.tensor(y, dtype=torch.float32)

device = 'cuda' if torch.cuda.is_available() else 'cpu' # 检测是否有 GPU

X = X.to(device)                                        # 将数据转移到 GPU
Y = Y.to(device)

# 定义数据集
class MyDataset(Dataset):
    def __init__(self,x,y):                             # 初始化数据集
        self.x = torch.tensor(x, dtype=torch.float32)   # 将数据转换为张量
        self.y = torch.tensor(y, dtype=torch.float32)   # 将数据转换为张量
    def __getitem__(self, index):                       # 根据索引返回数据
        return self.x[index], self.y[index]             
    def __len__(self):                                  # 返回数据集的长度
        return len(self.x)
    
ds = MyDataset(x,y)                                     # 创建数据集    
dl = DataLoader(ds, batch_size=2, shuffle=True)         # batch_size=2, 每次迭代返回两个样本

# 定义模型
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()         # 调用父类的构造函数
        self.input_to_hidden_layer = nn.Linear(2, 8)    # 输入层到隐藏层 
        self.hidden_layer_activation = nn.ReLU()        # 隐藏层激活函数
        self.hidden_to_output_layer = nn.Linear(8, 1)   # 隐藏层到输出层
        
    def forward(self, x):                               # 前向传播
        x = self.input_to_hidden_layer(x)               # 输入到隐藏层
        x = self.hidden_layer_activation(x)             # 隐藏层激活
        x = self.hidden_to_output_layer(x)              # 隐藏层到输出层
        return x
    
model = MyNeuralNetwork().to(device)                   # 创建模型并将其转移到 GPU

def my_loss(output, target):                           # 定义损失函数
    return ((output - target) ** 2).mean()              # 均方误差

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 定义优化器

loss_function = nn.MSELoss()                           # 使用 PyTorch 提供的均方误差

loss_value = loss_function(model(X), Y)                # 计算损失
print(loss_value)                                      # 打印损失
my_loss_value = my_loss(model(X), Y)                   # 计算自定义损失
print(my_loss_value)                                   # 打印自定义损失

#获取中间层的值
input_to_hidden_layer_output = model.input_to_hidden_layer(X)
hidden_activation_output = model.hidden_layer_activation(input_to_hidden_layer_output)
hidden_to_output_layer_output = model.hidden_to_output_layer(hidden_activation_output)
print(input_to_hidden_layer_output)
print(hidden_activation_output)
print(hidden_to_output_layer_output)

#中间层值一列表方式返回
class neuralnet(nn.Module):
    def __init__(self):
        super(neuralnet, self).__init__()
        self.input_to_hidden_layer = nn.Linear(2, 8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)
    def forward(self, x):
        hidden1 = self.input_to_hidden_layer(x)
        hidden2 = self.hidden_layer_activation(hidden1)
        output = self.hidden_to_output_layer(hidden2)
        return output, hidden1, hidden2
    
model2 = neuralnet().to(device)
print(model2(X)[0], model2(X)[1], model2(X)[2])