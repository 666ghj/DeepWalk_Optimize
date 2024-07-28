import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmbeddingLossFunctions(object):
    def __init__(self, loss_fn='xent', neg_sample_weights=1.0):
        """
        基础类，应用类似于skip-gram的损失
        （即，节点+目标和节点与负样本的点积）
        参数:
            loss_fn: 损失函数类型，默认为交叉熵（'xent'）
            neg_sample_weights: 负样本权重，默认为1.0
        """
        self.neg_sample_weights = neg_sample_weights
        self.output_dim = 1
        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss
        else:
            print("Not implemented yet.")

    def loss(self, inputs1, inputs2, neg_samples):
        """
        负采样损失
        参数:
            inputs1: 输入1
            inputs2: 输入2
            neg_samples: 负样本，形状为[num_neg_samples x input_dim2]
        """
        return self.loss_fn(inputs1, inputs2, neg_samples)

    def affinity(self, inputs1, inputs2):
        """
        计算输入1和输入2之间的亲和力
        参数:
            inputs1: 形状为[n_batch_edges x feature_size]的张量
            inputs2: 形状为[n_batch_edges x feature_size]的张量
        """
        result = torch.sum(inputs1 * inputs2, dim=1) # 形状: (n_batch_edges,)
        return result

    def neg_cost(self, inputs1, neg_samples):
        """
        对于每个输入，计算其与负样本的亲和力之和
        返回:
            形状为[n_batch_edges x num_neg_samples]的张量。对于每个节点，计算其与负样本的亲和力列表
        """
        neg_aff = inputs1.mm(neg_samples.t()) #(n_batch_edges, num_neg_samples)
        return neg_aff

    def sigmoid_cross_entropy_with_logits(self, labels, logits):
        """
        计算带有logits的sigmoid交叉熵
        参数:
            labels: 标签
            logits: logits
        """
        sig_aff = torch.sigmoid(logits)
        loss = labels * -torch.log(sig_aff) + (1 - labels) * -torch.log(1 - sig_aff)
        return loss

    def _xent_loss(self, inputs1, inputs2, neg_samples):
        """
        计算交叉熵损失
        参数:
            inputs1: 形状为(512, 256)的张量，归一化向量
            inputs2: 形状为(512, 256)的张量，归一化向量
            neg_samples: 形状为(20, 256)的张量
        """
        cuda = inputs1.is_cuda
        true_aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples)
        true_labels = torch.ones(true_aff.shape)  # (n_batch_edges,)
        if cuda:
            true_labels = true_labels.cuda()
        true_xent = self.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=true_aff)
        neg_labels = torch.zeros(neg_aff.shape)
        if cuda:
            neg_labels = neg_labels.cuda()
        neg_xent = self.sigmoid_cross_entropy_with_logits(labels=neg_labels, logits=neg_aff)
        loss0 = true_xent.sum()
        loss1 = self.neg_sample_weights * neg_xent.sum()
        loss = loss0 + loss1
        return loss, loss0, loss1


class MappingLossFunctions(object):
    def __init__(self):
        """
        初始化，设置损失函数为欧几里得损失
        """
        self.loss_fn = self._euclidean_loss

    def loss(self, inputs1, inputs2):
        """
        计算损失
        参数:
            inputs1: 输入1
            inputs2: 输入2
        """
        return self.loss_fn(inputs1, inputs2)

    def _euclidean_loss(self, inputs1, inputs2):
        """
        计算欧几里得损失
        参数:
            inputs1: 输入1
            inputs2: 输入2
        """
        sub = inputs2 - inputs1
        square_sub = sub**2
        loss = torch.sum(square_sub)        
        return loss