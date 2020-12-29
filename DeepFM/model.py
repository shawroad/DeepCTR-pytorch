"""
# -*- coding: utf-8 -*-
# @File    : model.py
# @Time    : 2020/12/28 7:02 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFM(nn.Module):
    def __init__(self, feature_sizes, embedding_size=4,
                 hidden_dims=[32, 32], num_classes=10, dropout=[0.5, 0.5]):
        '''
        :param feature_sizes: A list of integer giving the size of features for each field.
        :param embedding_size: An integer giving size of feature embedding.
        :param hidden_dims: A list of integer giving the size of each hidden layer.
        :param num_classes: 类别
        :param dropout: An integer giving size of instances used in each interation.
        '''
        super(DeepFM, self).__init__()

        self.field_size = len(feature_sizes)  # 这里的域个数就是原始所有特征的个数
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        # ###########FM模型走起#########
        # 前十三个连续值线性映射
        fm_first_order_linear = nn.ModuleList(
            [nn.Linear(feature_size, self.embedding_size) for feature_size in self.feature_sizes[:13]]
        )
        # 后面26个离散值词嵌入
        fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes[13:40]])

        # 连续值和离散值嵌入以后的结果
        self.fm_first_order_models = fm_first_order_linear.extend(fm_first_order_embeddings)

        fm_second_order_linear = nn.ModuleList(
            [nn.Linear(feature_size, self.embedding_size) for feature_size in self.feature_sizes[:13]]
        )
        fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes[13:40]]
        )
        self.fm_second_order_models = fm_second_order_linear.extend(fm_second_order_embeddings)

        # ###########Deep模型走起##########
        all_dims = [self.field_size * self.embedding_size] + self.hidden_dims + [self.num_classes]
        # 上面是为接下来的全连接每层的输入做准备  显然第一层就是所有特征词嵌入进行拼接
        # 接着进入我们预先定义的隐层维度
        # 最后 肯定就是类别了。

        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i-1], all_dims[i]))  # 全连接
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))   # 批量归一化
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i-1]))
        self.output = nn.Sigmoid()

    def forward(self, Xi, Xv):
        '''
        :param Xi: A tensor of input's index, shape of (N, field_size, 1)
        :param Xv: A tensor of input's value, shape of (N, field_size, 1)
        :return:
        '''
        # print(Xi.size())   # torch.Size([32, 39, 1])
        # print(Xv.size())   # torch.Size([32, 39])
        # FM部分
        # 一阶特征
        fm_first_order_emb_arr = []
        for i, emb in enumerate(self.fm_first_order_models):
            if i <= 12:
                # 连续值embedding
                Xi_tem = Xi[:, i, :]   # size: (32, 1)
                Xi_tem = Xi_tem.to(dtype=torch.float)
                fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())

            else:
                Xi_tem = Xi[:, i, :]
                Xi_tem = Xi_tem.to(dtype=torch.long)

                fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())

        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        # print(fm_first_order.size())   # torch.Size([32, 156])

        # 二阶交叉特征 use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
        fm_second_order_emb_arr = []
        for i, emb in enumerate(self.fm_second_order_models):
            if i <= 12:
                Xi_tem = Xi[:, i, :].to(dtype=torch.float)
                fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())
            else:
                Xi_tem = Xi[:, i, :].to(dtype=torch.long)
                fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())

        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5

        # Deep部分
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)   # 将词嵌入的部分维度全部拼接
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
        # print(fm_first_order.size())   # torch.Size([32, 156])
        # print(fm_second_order.size())  # torch.Size([32, 4])
        # print(deep_out.size())   # torch.Size([32, 32])

        total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + torch.sum(deep_out, 1)

        return self.output(total_sum)