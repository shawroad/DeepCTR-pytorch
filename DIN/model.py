"""
@file   : model.py
@time   : 2024-07-12
"""
import torch
import torch.nn as nn


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1,)))
        self.epsilon = 1e-9

    def forward(self, x):
        norm_x = (x - x.mean(dim=0)) / torch.sqrt(x.var(dim=0) + self.epsilon)
        p = torch.sigmoid(norm_x)
        x = self.alpha * x.mul(1 - p) + x.mul(p)
        return x


class ActivationUnit(nn.Module):
    def __init__(self, embedding_dim, dropout=0.2, fc_dims=[32, 16]):
        super(ActivationUnit, self).__init__()
        fc_layers = []

        input_dim = embedding_dim * 4
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(Dice())
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim

        fc_layers.append(nn.Linear(input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, query, user_behavior):
        # print(query.size())   # torch.Size([128, 1, 8])
        # print(user_behavior.size())   # torch.Size([128, 40, 8])

        seq_len = user_behavior.shape[1]    # 用户行为长度   40
        # queries = torch.cat([query] * seq_len, dim=1)
        queries = query.repeat(1, seq_len, 1)   # torch.Size([128, 40, 8])

        attn_input = torch.cat([queries, user_behavior,
                                queries - user_behavior,
                                queries * user_behavior], dim=-1)
        # queries: batch_size, 40, 8
        # user_behavior: batch_size, 40, 8
        # queries - user_behavior: batch_size, 40, 8
        # queries * user_behavior: batch_size, 40, 8

        # batch_size, 40, 32
        # print(attn_input.size())   # torch.Size([128, 40, 32])

        out = self.fc(attn_input)
        # print(out.size())   # torch.Size([128, 40, 1])
        return out


class AttentionPoolingLayer(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super(AttentionPoolingLayer, self).__init__()
        self.active_unit = ActivationUnit(embedding_dim=embedding_dim, dropout=dropout)

    def forward(self, query_ad, user_behavior, mask):
        # 待推商品 和 兴趣 融合的部分
        attns = self.active_unit(query_ad, user_behavior)  # Batch * seq_len * 1
        #  attns: torch.Size([128, 40, 1])

        # user_behavior: （batch_size, 40, 8）
        # attns： (batch_size, 40, 1)

        output = user_behavior.mul(attns.mul(mask))  # batch * seq_len * embed_dim
        # print(output.size())   # （batch_size, 40, 8）
        output = output.sum(dim=1)
        # print(output.size())   # (batch_size, 8）
        return output


class DeepInterestNet(nn.Module):
    def __init__(self, feature_dim, embed_dim, mlp_dims, dropout):
        super(DeepInterestNet, self).__init__()
        self.feature_dim = feature_dim

        # feature_dim: 所有的商品类别的总个数   +1   0
        self.embedding = nn.Embedding(feature_dim + 1, embed_dim)
        self.AttentionActivate = AttentionPoolingLayer(embed_dim, dropout)

        fc_layers = []
        # 由于只有用户历史行为和商品本身的ID,这里在embedding后concate后是2个embed size
        input_dim = embed_dim * 2
        for fc_dim in mlp_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim

        fc_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*fc_layers)

    def forward(self, x):
        # 行为特征40列  广告特征1列
        # print(x.size())    # torch.Size([128, 41])
        # 1. 通过切片只拿兴趣行为
        behaviors_x = x[:, :-1]   # 切片  只拿兴趣

        # 2. 因为一个batch_size中兴趣长度不一      [12, 32, 12, 0, 0, 0, 0, 0...]
        mask = (behaviors_x > 0).float().unsqueeze(-1)

        # 3. 推荐商品的id拿到
        ads_x = x[:, -1]

        # 4. 给推荐商品的类别做embedding
        query_ad = self.embedding(ads_x).unsqueeze(1)
        # print(query_ad.size())   # (batch_size,  1, 8)

        # 5. 对兴趣商品做embedding
        user_behavior = self.embedding(behaviors_x)
        # print(user_behavior.size())   # (batch_size, 40, 8)
        user_behavior = user_behavior.mul(mask)  # mul 对应位置元素直接相乘  消除padding的影响

        # 两个东西:
        # 一、query_ad 待推荐商品的embedding   (batch_size, 1, 8)
        # 二、user_behavior 用户的兴趣 embedding   (batch_size, 40, 8)

        # attention pooling layer
        user_interest = self.AttentionActivate(query_ad, user_behavior, mask)
        # print(user_interest.size())    # torch.Size([128, 8])

        concat_input = torch.cat([user_interest, query_ad.squeeze(1)], dim=1)
        # print(concat_input.size())   # torch.Size([64, 16])

        # MLP prediction
        out = self.mlp(concat_input)
        out = torch.sigmoid(out.squeeze(1))
        return out