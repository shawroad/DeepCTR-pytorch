"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-16
"""
import torch
from torch import nn
import numpy as np


class Interact_Layer(nn.Module):
    def __init__(self, embed_dim=8, n_heads=2, d=6):
        super(Interact_Layer, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d = d
        self.W_Q = nn.Linear(self.embed_dim, self.d * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.embed_dim, self.d * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.embed_dim, self.d * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d, self.embed_dim, bias=False)
        self.activate = nn.ReLU()

    def forward(self, input_q, input_k, input_v):
        residual, batch_size = input_q, input_q.size(0)
        Q = self.W_Q(input_q).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)
        K = self.W_K(input_k).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)
        V = self.W_V(input_v).view(batch_size, -1, self.n_heads, self.d).transpose(1, 2)
        # print(Q.size())   # torch.Size([2, 2, 39, 6])
        # print(K.size())   # torch.Size([2, 2, 39, 6])

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)   # [batch_size, n_heads, len_q, d_v] torch.Size([2, 2, 39, 6])
        multi_attention_output = context.transpose(1, 2).reshape(batch_size, -1, self.d * self.n_heads)
        multi_attention_output = self.fc(multi_attention_output)

        # 加入残差
        residual = input_q
        output = self.activate(multi_attention_output + residual)
        return output


class AutoInt(nn.Module):
    def __init__(self, cate_fea_uniques, num_fea_size=0, emb_size=8, n_layers=3):
        '''
        :param cate_fea_uniques:
        :param num_fea_size: 数字特征  也就是连续特征
        :param emb_size:
        '''
        super(AutoInt, self).__init__()
        self.cate_fea_size = len(cate_fea_uniques)
        self.num_fea_size = num_fea_size
        self.n_layers = n_layers

        self.sparse_embed = nn.ModuleList([nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_uniques])
        self.dense_embed = nn.ModuleList([nn.Linear(1, 8) for _ in range(self.num_fea_size)])
        self.interact_layer = Interact_Layer()
        self.output = nn.Linear((self.num_fea_size + self.cate_fea_size) * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """
        batch_size = X_sparse.size(0)
        # Sparse_feature
        sparse_kd_embed = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.sparse_embed)]
        sparse_embed_map = torch.cat(sparse_kd_embed, dim=1)   # torch.Size([2, 26, 8])

        # Dense_feature
        dense_kd_embed = [dense(X_dense[:, i].unsqueeze(1)).unsqueeze(1) for i, dense in enumerate(self.dense_embed)]
        dense_embed_map = torch.cat(dense_kd_embed, dim=1)   # torch.Size([2, 13, 8])

        embed_map = torch.cat((sparse_embed_map, dense_embed_map), dim=1)
        # print(embed_map.size())   # batch_size, all_features_num, emb_dim  torch.Size([2, 39, 8])
        for _ in range(self.n_layers):
            embed_map = self.interact_layer(embed_map, embed_map, embed_map)
        # print(embed_map.size())    # torch.Size([2, 39, 8])

        embed_map = embed_map.view(batch_size, -1)

        out = self.output(embed_map)
        out = self.sigmoid(out)
        return out

