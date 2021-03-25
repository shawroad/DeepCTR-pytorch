"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-24
"""
import torch
from torch import nn
import torch.nn.functional as F


class WideDeep(nn.Module):
    def __init__(self, cate_fea_uniques,
                 num_fea_size=0,
                 emb_size=8,
                 hidden_dims=[256, 128],
                 num_classes=1,
                 dropout=[0.2, 0.2]):
        '''
        :param cate_fea_uniques:
        :param num_fea_size: 数字特征  也就是连续特征
        :param emb_size:
        :param hidden_dims:
        :param num_classes:
        :param dropout:
        '''
        super(WideDeep, self).__init__()
        self.cate_fea_size = len(cate_fea_uniques)
        self.num_fea_size = num_fea_size
        self.n_layers = 3
        self.n_filters = 12
        self.k = emb_size

        # sparse特征二阶表示
        self.sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_uniques
        ])

        self.linear = nn.Linear(self.num_fea_size, 1)

        # DNN部分
        self.all_dims = [self.cate_fea_size * emb_size + self.num_fea_size] + hidden_dims

        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i-1]))

        self.output = nn.Linear(self.all_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """
        sparse_embed = [emb(X_sparse[:, i]) for i, emb in enumerate(self.sparse_emb)]
        sparse_embed = torch.cat(sparse_embed, dim=-1)   # batch_size, sparse_feature_num * emb_dim
        # print(sparse_embed.size())   # torch.Size([2, 208])

        x = torch.cat([sparse_embed, X_dense], dim=-1)
        # print(x.size())    # torch.Size([2, 221])

        """Wide部分"""
        wide_out = self.linear(X_dense)
        # print(wide_out.size())   # torch.Size([2, 1])

        """DNN部分"""
        dnn_out = x
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        # print(dnn_out.size())   # torch.Size([2, 128])
        deep_out = self.output(dnn_out)
        out = self.sigmoid(0.5 * wide_out + 0.5 * deep_out)
        return out