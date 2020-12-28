"""
# -*- coding: utf-8 -*-
# @File    : model.py
# @Time    : 2020/12/28 4:13 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import torch
import torch.nn as nn


class WideModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        # 就是一个全连接+激活
        super(WideModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.linear(x)
        return out


class DeepModel(nn.Module):
    def __init__(self, deep_columns_idx, embedding_columns_dict, hidden_size_list, dropouts, output_dim):
        '''
        :param deep_columns_idx: 输入到deep中的列。 eg:{'age': 0, 'gender':1, 'career':2...}
        :param embedding_columns_dict: dict include categories columns name and number of unique val and embedding dimension
            e.g. {'age':(10, 32),...}
        :param hidden_layers: 需要进行几层全连接
        :param dropouts:
        :param output_dim:
        '''
        super(DeepModel, self).__init__()
        self.embedding_columns_dict = embedding_columns_dict
        self.deep_columns_idx = deep_columns_idx

        # 对每个非连续特征建立词嵌入
        for key, val in embedding_columns_dict.items():
            setattr(self, 'dense_col_'+key, nn.Embedding(val[0], val[1]))

        # 所有特征进行词嵌入完以后的维度计算
        embedding_out = 0
        for col in self.deep_columns_idx.keys():
            if col in embedding_columns_dict:
                embedding_out += embedding_columns_dict[col][1]
            else:
                embedding_out += 1

        self.layers = nn.Sequential()
        hidden_size_list = [embedding_out] + hidden_size_list  # [embedding_out, 64, 32, 16]  是隐层的维度
        dropouts = [0.0] + dropouts

        for i in range(1, len(hidden_size_list)):
            self.layers.add_module(
                'hidden_layer_{}'.format(i-1),
                nn.Sequential(nn.Linear(hidden_size_list[i-1], hidden_size_list[i]),
                              nn.LeakyReLU(),
                              nn.Dropout(dropouts[i-1]))
            )
        self.layers.add_module('last_linear', nn.Linear(hidden_size_list[-1], output_dim))

    def forward(self, x):
        emb = []
        # 1. 不需要词嵌入的特征为连续特征  然后取出连续特征
        continuous_cols = [col for col in self.deep_columns_idx.keys() if col not in self.embedding_columns_dict]

        # 2. 对每个需要词嵌入(非连续特征)进行词嵌入
        for col, _ in self.embedding_columns_dict.items():
            idx = self.deep_columns_idx[col]
            emb.append(getattr(self, 'dense_col_' + col)(x[:, idx].long()))
        # print(emb[0].size())   # torch.Size([32, 8])
        # print(len(emb))  # 代表有四个特征需进行词嵌入
        # 3. 把连续特征也加进来
        for col in continuous_cols:
            idx = self.deep_columns_idx[col]
            emb.append(x[:, idx].view(-1, 1).float())
        # print(len(emb))   # 7  说明还有三个连续的特征需要加入
        # 此时emb:[torch.Size([32, 8])有四个， torch.Size([32, 1])有三个）
        embedding_dim = torch.cat(emb, dim=1)
        # print(embedding_dim.size())   # torch.Size([32, 35])
        out = self.layers(embedding_dim)
        return out


class WideDeep(nn.Module):
    def __init__(self, wide_model_params, deep_model_params):
        super(WideDeep, self).__init__()
        # 实例化wide模型
        wide_input_dim = wide_model_params['wide_input_dim']
        wide_output_dim = wide_model_params['wide_output_dim']  # 1
        self.wide = WideModel(wide_input_dim, wide_output_dim)

        # 实例化deep模型
        deep_columns_idx = deep_model_params['deep_columns_idx']   # 输入到deep模型中的列
        embedding_columns_dict = deep_model_params['embedding_columns_dict']    # 要进行词嵌入的列
        hidden_size_list = deep_model_params['hidden_size_list']
        dropouts = deep_model_params['dropouts']
        deep_output_dim = deep_model_params['deep_output_dim']
        self.deep = DeepModel(deep_columns_idx=deep_columns_idx,
                              embedding_columns_dict=embedding_columns_dict,
                              hidden_size_list=hidden_size_list,
                              dropouts=dropouts,
                              output_dim=deep_output_dim)
        self.output = nn.Sigmoid()

    def forward(self, x):
        """
        input and forward
        :param x: tuple(wide_model_data, deep_model_data, target)
        :return:
        """
        # wide model
        wide_data = x[0]
        wide_out = self.wide(wide_data.float())
        # print(wide_out.size())   # torch.Size([32, 1])

        # deep model
        deep_data = x[1]
        deep_out = self.deep(deep_data)
        # print(deep_out.size())   # torch.Size([32, 1])
        assert wide_out.size() == deep_out.size()

        wide_deep = wide_out.add(deep_out)   # wide的数据+deep的数据
        # print(wide_deep.size())   # torch.Size([32, 1])
        return self.output(wide_deep)
