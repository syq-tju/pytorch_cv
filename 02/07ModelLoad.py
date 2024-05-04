import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = 'mymodel.pth'

import os
file_size = os.path.getsize(save_path)
print(f"Size of {save_path}: {file_size / 1024:.2f} KB")

model2 = nn.Sequential(     # 创建一个新的模型
    nn.Linear(2,8),
    nn.ReLU(),
    nn.Linear(8,1)
    ).to(device)

model2.load_state_dict(torch.load(save_path))   # 加载模型参数
print(model2.state_dict() )