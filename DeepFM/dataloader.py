"""
# -*- coding: utf-8 -*-
# @File    : dataloader.py
# @Time    : 2020/12/28 7:55 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class CriteoDataset(Dataset):
    def __init__(self, root='data', train=True):
        '''
        :param root: 根目录
        :param train:
        '''
        self.root = root
        self.train = train

        if self.train:
            data = pd.read_csv(os.path.join(root, 'train.txt'))
            self.train_data = data.iloc[:, :-1].values
            self.target = data.iloc[:, -1].values
        else:
            data = pd.read_csv(os.path.join(root, 'test.txt'))
            self.test_data = data.iloc[:, :-1].values

    def __getitem__(self, idx):
        if self.train:
            dataI, targetI = self.train_data[idx, :], self.target[idx]
            Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)
            Xv = torch.from_numpy(np.ones_like(dataI))
            return Xi, Xv, targetI
        else:
            dataI = self.test_data.iloc[idx, :]
            Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)
            Xv = torch.from_numpy(np.ones_like(dataI))
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

