from loss import MappingLossFunctions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##############################################################################
#               MAPPING MODELS
##############################################################################

class PaleMapping(nn.Module):
    def __init__(self, source_embedding, target_embedding):
        """
        参数
        ----------
        source_embedding: torch.Tensor 或 Embedding_model
            用于获取节点的嵌入向量
        target_embedding: torch.Tensor 或 Embedding_model
            用于获取目标节点的嵌入向量
        target_neighbor: 字典
            目标节点 -> 目标节点的邻居的字典。用于计算vinh_loss
        """

        super(PaleMapping, self).__init__()
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.loss_fn = MappingLossFunctions() # 这里计算欧几里得损失


class PaleMappingMlp(PaleMapping):
    def __init__(self, embedding_dim, source_embedding, target_embedding, activate_function='relu'):
        """
        参数
        ----------
        embedding_dim: int
            嵌入向量的维度
        source_embedding: torch.Tensor 或 Embedding_model
            用于获取节点的嵌入向量
        target_embedding: torch.Tensor 或 Embedding_model
            用于获取目标节点的嵌入向量
        activate_function: str
            激活函数的类型，可以是 'sigmoid'、'relu' 或 'tanh'
        """

        super(PaleMappingMlp, self).__init__(source_embedding, target_embedding)

        if activate_function == 'sigmoid':
            self.activate_function = nn.Sigmoid()
        elif activate_function == 'relu':
            self.activate_function = nn.ReLU()
        else:
            self.activate_function = nn.Tanh()
            
        hidden_dim = 2*embedding_dim
        # 定义一个多层感知机，包含两个全连接层和一个激活函数
        self.mlp = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim, bias=True), # bias=True表示包含偏置
            self.activate_function,
            nn.Linear(hidden_dim, embedding_dim, bias=True)
        ])
        """
                    [输入层]                例：4维
                        ↓
                    [全连接层1]             例：8维
                        |   (包含偏置)
                        ↓
                    [激活函数]              例：relu
                        ↓
                    [全连接层2]             例：4维
                        |   (包含偏置)
                        ↓
                    [输出层]                例：4维
        """


    def loss(self, source_indices, target_indices):
        """
        计算损失函数
        参数
        ----------
        source_indices: list 或 torch.Tensor
            源节点的索引
        target_indices: list 或 torch.Tensor
            目标节点的索引
        """
        # 获取源节点和目标节点的嵌入向量
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]
        # 将源节点的嵌入向量通过多层感知机进行映射
        source_feats_after_mapping = self.forward(source_feats)

        # source_indices、target_indices可以是一个列表，例如，假设self.source_embedding的形状是(1000, 128)，source_indices是[0, 1,  2]，那么source_feats的形状就是(3, 128)，也就是说，它包含了3个128维的嵌入向量。
        # 那么shape[0]就是3，也就是batch_size，表示了有多少个样本数量
        batch_size = source_feats.shape[0]

        # 计算映射后的源节点嵌入向量和目标节点嵌入向量之间的损失
        mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats) / batch_size

        return mapping_loss

    def forward(self, source_feats):
        """
        前向传播
        参数
        ----------
        source_feats: torch.Tensor
            源节点的嵌入向量
        """
        # 将源节点的嵌入向量通过多层感知机进行映射
        ret = self.mlp(source_feats)
        # 对映射后的嵌入向量进行归一化
        # 归一化的主要目的是消除向量长度的影响，使得不同的向量可以在同一尺度下进行比较
        ret = F.normalize(ret, dim=1)
        return ret