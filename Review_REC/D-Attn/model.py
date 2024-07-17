"""
@file   : model.py
@time   : 2024-07-16
"""

import torch
import torch.nn as nn
from config import set_args
import torch.nn.functional as F

args = set_args()


class LocalAttention(nn.Module):
    def __init__(self, seq_len, win_size, emb_size, filters_num):
        super(LocalAttention, self).__init__()
        self.att_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(win_size, emb_size), padding=((win_size-1)//2, 0)),
            nn.Sigmoid()
        )
        self.cnn = nn.Conv2d(1, filters_num, kernel_size=(1, emb_size))

    def forward(self, x):
        # print(x.size())   #  torch.Size([64, 10, 300])
        score = self.att_conv(x.unsqueeze(1)).squeeze(1)
        # print(score.size())   #  torch.Size([64, 10, 1])
        out = x.mul(score)

        out = out.unsqueeze(1)   # torch.Size([64, 1, 10, 300])
        out = torch.tanh(self.cnn(out)).squeeze(3)
        # print(out.size())    #  torch.Size([64, 100, 10])
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        # print(out.size())    #  torch.Size([64, 100])
        return out


class GlobalAttention(nn.Module):
    def __init__(self, seq_len, emb_size, filters_size=[2, 3, 4], filters_num=100):
        super(GlobalAttention, self).__init__()
        self.att_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(seq_len, emb_size)),
            nn.Sigmoid()
        )
        self.convs = nn.ModuleList([nn.Conv2d(1, filters_num, (k, emb_size)) for k in filters_size])

    def forward(self, x):
        x = x.unsqueeze(1)
        score = self.att_conv(x)
        x = x.mul(score)
        conv_outs = [torch.tanh(cnn(x).squeeze(3)) for cnn in self.convs]
        conv_outs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]
        return conv_outs


class Net(nn.Module):
    def __init__(self, word_emb):
        super(Net, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        emb_size = self.embedding.embedding_dim
        self.local_att = LocalAttention(args.review_count, win_size=5, emb_size=emb_size, filters_num=args.filters_num)
        self.global_att = GlobalAttention(args.review_count, emb_size=emb_size, filters_num=args.filters_num)

        fea_dim = args.filters_num * 4
        self.fc = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(fea_dim, args.id_emb_size),
        )
        self.dropout = nn.Dropout(0.5)
        self.reset_para()

    def forward(self, docs):
        docs = self.embedding(docs)  # size * 300
        docs = docs.sum(dim=-2)  # output(batch_size, review_count, word_dim)
        local_fea = self.local_att(docs)   # torch.Size([64, 100])

        global_fea = self.global_att(docs)
        r_fea = torch.cat([local_fea]+global_fea, 1)
        r_fea = self.dropout(r_fea)
        r_fea = self.fc(r_fea)
        return torch.stack([r_fea], 1)

    def reset_para(self):
        cnns = [self.local_att.cnn, self.local_att.att_conv[0]]
        for cnn in cnns:
            nn.init.xavier_uniform_(cnn.weight, gain=1)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)
        for cnn in self.global_att.convs:
            nn.init.xavier_uniform_(cnn.weight, gain=1)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc[0].weight, -0.1, 0.1)
        nn.init.uniform_(self.fc[-1].weight, -0.1, 0.1)


class FactorizationMachine(nn.Module):
    def __init__(self, in_dim, k):
        super(FactorizationMachine, self).__init__()
        self.v = nn.Parameter(torch.zeros(2 * in_dim, k))
        self.linear = nn.Linear(2 * in_dim, 1)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, in_dim), output shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v)
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 ** 2 - inter_part2, dim=1)
        output = linear_part.t() + 0.5 * pair_interactions
        return output.view(-1, 1)  # output shape(batch_size, 1)


class D_ATTN(nn.Module):
    def __init__(self, word_emb):
        super(D_ATTN, self).__init__()
        self.user_net = Net(word_emb)
        self.item_net = Net(word_emb)
        self.fm = FactorizationMachine(in_dim=args.id_emb_size, k=args.fm_hidden)

    def forward(self, user_reviews, item_reviews):
        u_fea = self.user_net(user_reviews)
        i_fea = self.item_net(item_reviews)
        # print(u_fea.size())   #
        # print(i_fea.size())   #
        i_fea = i_fea.squeeze(1)
        u_fea = u_fea.squeeze(1)
        # print(u_fea.size())   # torch.Size([64, 32])
        # print(i_fea.size())   #  torch.Size([64, 32])

        prediction = self.fm(torch.cat([u_fea, i_fea], dim=-1))
        return prediction
