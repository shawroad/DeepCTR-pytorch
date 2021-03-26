"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-23
"""
import torch
from torch import nn
import torch.nn.functional as F


class PNN(nn.Module):
    def __init__(self, cate_fea_uniques, num_fea_size=0, emb_size=8,
                 hidden_dims=[256, 128, 64],
                 dropout=[0.2, 0.2],
                 num_classes=1,
                 mode='out'):
        '''
        :param cate_fea_uniques:
        :param num_fea_size: 数字特征  也就是连续特征
        :param emb_size: embed_dim
        '''
        super(PNN, self).__init__()
        self.cate_fea_size = len(cate_fea_uniques)
        self.num_fea_size = num_fea_size
        self.emb_size = emb_size
        self.mode = mode

        self.embed_layers = nn.ModuleList([
            nn.Embedding(voc_size, self.emb_size) for voc_size in cate_fea_uniques
        ])

        if self.mode == 'in':
            self.w_p = torch.randn((self.cate_fea_size * (self.cate_fea_size - 1) // 2,
                                    self.emb_size,
                                    hidden_dims[0]), requires_grad=True)

        else:
            self.w_p = torch.randn((self.cate_fea_size * (self.cate_fea_size - 1) // 2,
                                   self.emb_size,
                                   self.emb_size,
                                   hidden_dims[0]), requires_grad=True)

        # parameters
        self.w_z = torch.randn((self.cate_fea_size * self.emb_size, hidden_dims[0]),
                               requires_grad=True)
        self.l_b = torch.randn((hidden_dims[0], ), requires_grad=True)

        self.all_dims = hidden_dims
        self.all_dims[0] = self.all_dims[0] + self.num_fea_size
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

        # product layer
        p = []
        q = []
        for i in range(self.cate_fea_size - 1):
            for j in range(i + 1, self.cate_fea_size):
                p.append(embed[:, i, :].unsqueeze(1))
                q.append(embed[:, j, :].unsqueeze(1))
        p = torch.cat(p, dim=1)   # torch.Size([2, 325, 8])
        q = torch.cat(q, dim=1)   # torch.Size([2, 325, 8])

        batch_size = p.size(0)
        if self.mode == 'in':
            # temp = p*q
            # print(temp.size())   #torch.Size([2, 325, 8])
            # print(self.w_p.size())   # torch.Size([325, 8, 256])
            temp = (p * q).view(batch_size, -1)
            self.w_p = self.w_p.view(temp.size(1), -1)
            l_p = torch.matmul(temp, self.w_p)
        else:
            u = p.unsqueeze(2)   # torch.Size([2, 325, 1, 8])
            v = q.unsqueeze(2)   # torch.Size([2, 325, 1, 8])
            u = u.permute(0, 1, 3, 2)   # torch.Size([2, 325, 8, 1])
            temp = torch.matmul(u, v)   # torch.Size([2, 325, 8, 8])
            temp = temp.view(batch_size, -1)
            self.w_p = self.w_p.view(temp.size(1), -1)
            l_p = torch.matmul(temp, self.w_p)

        l_z = torch.matmul(embed.view(batch_size, -1), self.w_z)

        dnn_out = F.relu(torch.cat([l_z + l_p + self.l_b, X_dense], dim=-1))
        # print(dnn_out.size())   # torch.Size([2, 269])
        # dnn
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)
        out = self.dnn_linear(dnn_out)   # batch_size, 1
        out = self.sigmoid(out)
        return out
