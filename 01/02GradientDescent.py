import numpy as np
from copy import deepcopy

# 之前定义的前向传播函数
def feed_forward(inputs, outputs, weights):       
    pre_hidden = np.dot(inputs, weights[0]) + weights[1]
    hidden = 1 / (1 + np.exp(-pre_hidden))
    pred_out = np.dot(hidden, weights[2]) + weights[3]
    mean_squared_error = np.mean(np.square(pred_out - outputs))
    return mean_squared_error

def update_weights(inputs, outputs, weights, lr):
    original_weights = deepcopy(weights)
    temp_weights = deepcopy(weights)
    updated_weights = deepcopy(weights)
    original_loss = feed_forward(inputs, outputs, original_weights)
    for i, layer in enumerate(original_weights):
        for index, weight in np.ndenumerate(layer):
            temp_weights = deepcopy(weights)
            temp_weights[i][index] += 0.0001
            _loss_plus = feed_forward(inputs, outputs, temp_weights)
            grad = (_loss_plus - original_loss)/(0.0001)
            updated_weights[i][index] -= grad*lr
    return updated_weights, original_loss

# 初始化数据
inputs = np.array([[0.5, 1.5],
                   [1.0, -0.5],
                   [0.0, 1.0]])
outputs = np.array([[1.2],
                    [0.5],
                    [0.8]])
weights = [np.random.rand(2, 2),  # 输入层到隐藏层的权重
           np.random.rand(2),     # 隐藏层的偏置
           np.random.rand(2, 1),  # 隐藏层到输出层的权重
           np.random.rand(1)]     # 输出层的偏置
lr = 0.01

# 更新权重并计算原始损失
updated_weights, original_loss = update_weights(inputs, outputs, weights, lr)

print(f"Original Loss: {original_loss}")
print(f"Updated Weights: {updated_weights}")
