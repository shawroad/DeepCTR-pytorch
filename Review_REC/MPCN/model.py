"""
@file   : model.py
@time   : 2024-07-16
"""
import torch
from torch import nn
from config import set_args

args = set_args()

import torch
from torch import nn


class GatingMachine(nn.Module):
    def __init__(self, word_dim):
        super(GatingMachine, self).__init__()
        self.gating1 = nn.Sequential(
            nn.Linear(in_features=word_dim, out_features=word_dim, bias=False),
            nn.Sigmoid()
        )
        self.gating2 = nn.Sequential(
            nn.Linear(in_features=word_dim, out_features=word_dim, bias=True),
            nn.Tanh()
        )
        self.bg = nn.Parameter(torch.randn(word_dim))

    def forward(self, x):
        out1 = self.gating1(x)
        out2 = self.gating2(x)
        bg = self.bg.expand_as(out2)
        out = out1 + bg * out2
        return out


class CoAttention(nn.Module):
    def __init__(self, word_dim, mode='max', gumbel=True):
        super(CoAttention, self).__init__()
        self.mode = mode
        self.gumbel = gumbel
        self.linear = nn.Sequential(
            nn.Linear(in_features=word_dim, out_features=word_dim),
            nn.ReLU()
        )
        self.M = nn.Parameter(torch.randn(word_dim, word_dim))

    def forward(self, user_reviews, item_reviews):
        u_outs = self.linear(user_reviews)
        i_outs = self.linear(item_reviews)
        s = u_outs @ self.M.expand(u_outs.shape[0], -1, -1) @ i_outs.transpose(-1, -2)  # Affinity Matrix
        if self.mode == 'max':
            pooling_row = s.max(dim=-1).values
            pooling_col = s.max(dim=-2).values
        elif self.mode == 'mean':
            pooling_row = s.mean(dim=-1)
            pooling_col = s.mean(dim=-2)
        else:
            print('Please correct the parameter "mode" when defining the CoAttention!')
            print('It Supports "max" and "mean".')
            pooling_row, pooling_col = None, None
            exit(0)

        if self.gumbel:
            pooling_row = nn.functional.gumbel_softmax(pooling_row, tau=0.5, hard=True, dim=-1)
            pooling_col = nn.functional.gumbel_softmax(pooling_col, tau=0.5, hard=True, dim=-1)  # return one-hot vector
        else:
            pooling_row = nn.functional.softmax(pooling_row, dim=-1)
            pooling_col = nn.functional.softmax(pooling_col, dim=-1)  # return probability distribution
        return pooling_row, pooling_col


class FactorizationMachine(nn.Module):

    def __init__(self, in_dim, k):
        super(FactorizationMachine, self).__init__()
        self.v = nn.Parameter(torch.zeros(in_dim, k))
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, in_dim), output shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v)
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 ** 2 - inter_part2, dim=1)
        output = linear_part.t() + 0.5 * pair_interactions
        return output.view(-1, 1)  # output shape(batch_size, 1)


class MPCN(nn.Module):

    def __init__(self, word_emb, fusion_mode='concatenate'):
        super(MPCN, self).__init__()
        self.pointer_count = args.pointer_count
        self.fusion_mode = fusion_mode
        # Input Encoding
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.word_dim = self.embedding.embedding_dim
        self.gating = GatingMachine(self.word_dim)
        # Review-level Co-Attention & Word-level Co-Attention
        self.review_atte = CoAttention(self.word_dim, mode='max', gumbel=True)
        self.word_atte = CoAttention(self.word_dim, mode='mean', gumbel=False)

        if fusion_mode == 'concatenate':
            self.fm = FactorizationMachine(in_dim=2 * self.pointer_count * self.word_dim, k=args.fm_hidden)
        elif fusion_mode == 'sum':
            self.fm = FactorizationMachine(in_dim=2 * self.word_dim, k=args.fm_hidden)
        elif fusion_mode == 'linear':
            self.u_linear = nn.Sequential(
                nn.Linear(2 * self.pointer_count * self.word_dim, self.word_dim),
                nn.ReLU()
            )
            self.i_linear = nn.Sequential(
                nn.Linear(2 * self.pointer_count * self.word_dim, self.word_dim),
                nn.ReLU()
            )
            self.fm = FactorizationMachine(in_dim=2 * self.word_dim, k=args.fm_hidden)


    def forward(self, user_reviews, item_reviews):  # input(batch_size, review_count, review_length)
        # Input Encoding
        u_reviews_emb = self.embedding(user_reviews)  # output(batch_size, review_count, review_length, word_dim)
        i_reviews_emb = self.embedding(item_reviews)

        u_reviews = u_reviews_emb.sum(dim=-2)  # output(batch_size, review_count, word_dim)
        i_reviews = i_reviews_emb.sum(dim=-2)
        u_reviews = self.gating(u_reviews)
        i_reviews = self.gating(i_reviews)
        # Co-Attention
        u_repr, i_repr = [], []
        for i in range(self.pointer_count):
            # Review-level Co-Attention
            u_r_one_hot, i_r_one_hot = self.review_atte(u_reviews, i_reviews)
            user_review = u_r_one_hot.unsqueeze(-1).unsqueeze(-1) * u_reviews_emb
            item_review = i_r_one_hot.unsqueeze(-1).unsqueeze(-1) * i_reviews_emb
            user_review = user_review.sum(dim=-3)
            item_review = item_review.sum(dim=-3)  # output(batch_size, review_length, word_dim)
            # Word-level Co-Attention
            u_w_atte, i_w_atte = self.word_atte(user_review, item_review)
            u_w_repr = u_w_atte.unsqueeze(-2) @ user_review  # Weighted words
            i_w_repr = i_w_atte.unsqueeze(-2) @ item_review
            u_repr.append(u_w_repr.squeeze(-2))
            i_repr.append(i_w_repr.squeeze(-2))
        # Multi-Pointer Fusion
        if self.fusion_mode == 'concatenate':
            u_repr = torch.cat(u_repr, dim=-1)
            i_repr = torch.cat(i_repr, dim=-1)
        elif self.fusion_mode == 'sum':
            u_repr = torch.sum(torch.stack(u_repr, dim=0), dim=0)
            i_repr = torch.sum(torch.stack(i_repr, dim=0), dim=0)
        elif self.fusion_mode == 'linear':
            u_repr = torch.cat(u_repr, dim=-1)
            i_repr = torch.cat(i_repr, dim=-1)
            u_repr = self.u_linear(u_repr)
            i_repr = self.i_linear(i_repr)

        prediction = self.fm(torch.cat([u_repr, i_repr], dim=-1))
        return prediction