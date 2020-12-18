"""
# -*- coding: utf-8 -*-
# @File    : model.py
# @Time    : 2020/12/17 3:18 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import torch
from torch import nn
import numpy as np


class FeaturesLinear(nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super(FeaturesLinear, self).__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)), requires_grad=True)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # print(self.offsets.shape)    # (22,)
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FactorizationMachine(nn.Module):
    def __init__(self, reduce_sum=True):
        super(FactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FactorizationMachineModel(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super(FactorizationMachineModel, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # 1. 域嵌入
        embed = self.embedding(x)
        # print(embed.size())   # torch.Size([32, 22, 16])
        # fm
        fm_output = self.fm(embed)
        # 线性模型
        linear_out = self.linear(x)
        # print(fm_output.size())  # torch.Size([32, 1])
        # print(linear_out.size())  # torch.Size([32, 1])

        output = fm_output + linear_out
        # print(output.size())   #  torch.Size([32, 1])
        return torch.sigmoid(output.squeeze(1))
