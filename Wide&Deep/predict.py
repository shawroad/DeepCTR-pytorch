"""
# -*- coding: utf-8 -*-
# @File    : predict.py
# @Time    : 2020/12/28 4:13 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import torch
from model import WideDeep
from data_process import read_data, feature_engine
from config import set_args

args = set_args()
path = './data/adult.data'
data = read_data(path)
train_data, test_data, deep_columns_idx, embedding_columns_dict = feature_engine(data)
data_wide = train_data[0]

# 预测数据的输入格式，这里预测一条数据
t = (torch.from_numpy(train_data[0].values[0].reshape(-1, train_data[0].values.shape[1])),
     torch.from_numpy(train_data[1].values[0].reshape(-1, train_data[1].values.shape[1])))

# parameters setting
deep_model_params = {
    'deep_columns_idx': deep_columns_idx,
    'embedding_columns_dict': embedding_columns_dict,
    'hidden_size_list': args.hidden_size_list,
    'dropouts': args.dropouts,
    'deep_output_dim': args.deep_out_dim}
wide_model_params = {
    'wide_input_dim': data_wide.shape[1],
    'wide_output_dim': args.wide_out_dim
}
model = WideDeep(wide_model_params, deep_model_params)
# path 为存储模型参数的位置
path = 'wide_deep_model_0.pkl'

model.load_state_dict(torch.load(path))
print('输出的结果:', int(model(t) > 0.5))
