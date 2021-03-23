"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-23
"""
import torch
from torch import nn


class NFM(nn.Module):
    def __init__(self, cate_fea_uniques, num_fea_size=0, emb_size=8,
                 hidden_dims=[256, 128],
                 dropout=[0.2, 0.2],
                 num_classes=1):
        '''
        :param cate_fea_uniques:
        :param num_fea_size: 数字特征  也就是连续特征
        :param emb_size: embed_dim
        '''
        super(NFM, self).__init__()
        self.cate_fea_size = len(cate_fea_uniques)
        self.num_fea_size = num_fea_size
        self.emb_size = emb_size

        self.embed_layers = nn.ModuleList([
            nn.Embedding(voc_size, self.emb_size) for voc_size in cate_fea_uniques
        ])

        self.all_dims = [self.emb_size + self.num_fea_size] + hidden_dims

        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i-1]))

        self.dnn_linear = nn.Linear(hidden_dims[-1], num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """
        embed = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.embed_layers)]
        embed = torch.cat(embed, dim=1)  # torch.Size([2, 26, 8])  batch_size, cat_num, hidden_size

        # bi-interaction layer
        embed = 0.5 * (torch.pow(torch.sum(embed, dim=1), 2)) - torch.sum(torch.pow(embed, 2), dim=1)
        # print(embed.size())  # (batch_size, hidden_size)

        # concat
        dnn_out = torch.cat([X_dense, embed], dim=-1)
        # print(dnn_out.size())    # torch.Size([2, 21])
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        out = self.dnn_linear(dnn_out)   # batch_size, 1
        out = self.sigmoid(out)
        return out
