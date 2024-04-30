import numpy as np

def feed_forward(inputs, outputs, weights):       
    pre_hidden = np.dot(inputs,weights[0])+ weights[1]
    hidden = 1/(1+np.exp(-pre_hidden))
    pred_out = np.dot(hidden, weights[2]) + weights[3]
    mean_squared_error = np.mean(np.square(pred_out - outputs))
    return mean_squared_error

# 定义输入数据（假设有3个样本，每个样本有2个特征）
inputs = np.array([[0.5, 1.5],
                   [1.0, -0.5],
                   [0.0, 1.0]])

# 定义真实输出数据（假设每个样本有1个输出）
outputs = np.array([[1.2],
                    [0.5],
                    [0.8]])

# 定义权重和偏置
weights = [np.array([[0.2, 0.8],   # 输入到隐藏层的权重
                     [-0.5, 0.1]]),
           np.array([0.5, -0.3]),  # 隐藏层的偏置
           np.array([[1.0],        # 隐藏层到输出层的权重
                     [0.3]]),
           np.array([0.2])]        # 输出层的偏置

# 调用函数计算均方误差
mse = feed_forward(inputs, outputs, weights)
print(f"Mean Squared Error: {mse}")
