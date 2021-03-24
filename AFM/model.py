"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-23
"""
import torch
import itertools
from torch import nn
import torch.nn.functional as F


class AFM(nn.Module):
    def __init__(self, cate_fea_uniques, num_fea_size=0, emb_size=8,
                 num_classes=1,
                 mode='max'):
        '''
        :param cate_fea_uniques:
        :param num_fea_size: 数字特征  也就是连续特征
        :param emb_size: embed_dim
        '''
        super(AFM, self).__init__()
        self.cate_fea_size = len(cate_fea_uniques)
        self.num_fea_size = num_fea_size
        self.emb_size = emb_size
        self.mode = mode

        self.embed_layers = nn.ModuleList([
            nn.Embedding(voc_size, self.emb_size) for voc_size in cate_fea_uniques
        ])

        self.attention_W = nn.Linear(in_features=emb_size, out_features=8)
        self.attention_dense = nn.Linear(in_features=8, out_features=1)

        self.dnn_linear = nn.Linear(self.num_fea_size + emb_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """
        embed = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.embed_layers)]
        embed = torch.cat(embed, dim=1)  # torch.Size([2, 26, 8])  batch_size, cat_num, hidden_size

        # Pair-wise Interaction Layer
        row, col = [], []
        p = []
        q = []
        for r, c in itertools.combinations(range(self.cate_fea_size), 2):
            p.append(embed[:, r, :].unsqueeze(1))
            q.append(embed[:, c, :].unsqueeze(1))
        p = torch.cat(p, dim=1)
        q = torch.cat(q, dim=1)

        bi_interaction = p * q   # batch_size, (cat_num*cat_num-1)/2, embed_dim

        # mode
        if self.mode == 'max':
            x = torch.sum(bi_interaction, dim=1)
        elif self.mode == 'avg':
            x = torch.mean(bi_interaction, dim=1)
        else:
            x = self.attention(bi_interaction)

        x = torch.cat((x, X_dense), dim=-1)
        x = self.dnn_linear(x)
        outputs = self.sigmoid(x)
        return outputs

    def attention(self, bi_interaction):
        a = self.attention_W(bi_interaction)
        a = self.attention_dense(a)
        a_score = F.softmax(a, dim=1)
        outputs = torch.sum(bi_interaction * a_score, dim=1)
        return outputs

