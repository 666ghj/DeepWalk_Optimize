import torch
from torch.optim import Adam
import numpy as np
from mapping_model import PaleMappingMlp

# 读取源嵌入向量和目标嵌入向量
source_embedding = torch.tensor(np.loadtxt('source_embedding.txt')).float()
target_embedding = torch.tensor(np.loadtxt('target_embedding.txt')).float()

# 创建映射模型
model = PaleMappingMlp(embedding_dim=128, source_embedding=source_embedding, target_embedding=target_embedding, activate_function='relu')

# 创建优化器
optimizer = Adam(model.parameters())

# 训练模型
for epoch in range(100):  # 这里假设我们训练100个epoch
    # 随机选择一个批次的数据
    indices = torch.randperm(source_embedding.size(0))[:32]  # 这里假设批次大小为32
    # 计算损失
    loss = model.loss(indices, indices)
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    # 清空梯度
    optimizer.zero_grad()

# 使用模型进行映射
source_indices = torch.tensor([0, 1, 2])  # 这里假设我们要映射第0、1、2个源节点
mapped_source_embedding = model.forward(source_embedding[source_indices])