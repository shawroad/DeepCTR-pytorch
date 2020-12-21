"""
# -*- coding: utf-8 -*-
# @File    : dataloader.py
# @Time    : 2020/12/18 4:03 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import shutil
import struct
from collections import defaultdict
from pathlib import Path
import lmdb
import numpy as np
from torch.utils.data import Dataset


class AvazuDataset(Dataset):
    """
    删除不频繁的特征,将他们视为单个特征
    """
    def __init__(self, dataset_path=None, cache_path='.avazu', rebuild_cache=False, min_threshold=4):
        '''
        :param dataset_path: avazu train path
        :param cache_path: lmdb cache path
        :param rebuild_cache: If True, lmdb cache is refreshed
        :param min_threshold: 衡量频繁值的阈值
        '''
        self.NUM_FEATS = 22
        self.min_threshold = min_threshold

        if rebuild_cache or not Path(cache_path).exists():
            # 这里是预处理数据
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)

        # 加载数据
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
            # 特征数据, 标签
        return np_array[1:], np_array[0]

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        # 两个路径 一个是数据路径 一个是处理完数据保存的位置
        feat_mapper, defaults = self.__get_feat_mapper(path)

        # lmdb可以相乘key-value的数据库  map_size表示存储的最大容量
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                field_dims[i - 1] = len(fm) + 1
            # print(field_dims)   # 相当于每个特征下取值的个数加1就是field_dims
            # [  5   6   3 117  98  13  71  18  11   3  63 257   5   5 197   4   5 123
            #    5  37  63  30]
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())

            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __get_feat_mapper(self, path):
        '''
        过滤特征下低频取值， 将低于阈值的低频值 可以想成语言模型中的unk
        :param path:
        :return:
        feat_mapper: {1: {'14102221': 0, '14102100': 1, '14102706': 2, '14102816': 3}, 2: {'1002': 0, }, ...}
        defaults: {1: 4, 2: 5, 3: 2, 4: 116, 5: 97, 6: 12, 7: 70, 8: 17, 9: 10, 10: 2,
        '''
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            for line in f.readlines():
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    # 数据格式不正确
                    continue
                for i in range(1, self.NUM_FEATS + 1):
                    # id不要, click不要
                    feat_cnts[i][values[i + 1]] += 1    # 统计每个特征下  每个取值分别出现了多少次

        # 过滤低频的特征
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}

        # 接下来对特征进行数字映射
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        # print(feat_mapper)
        # {1: {'14102221': 0, '14102100': 1, '14102706': 2, '14102816': 3}, 2: {'1002': 0, '1005': 1, '1012': 2}, ...}

        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        # 相当于制作unknown的编码吧
        # 下面一行代表的意思: 第一个特征 能取4个值, 第二特征  能取5个值....
        # {1: 4, 2: 5, 3: 2, 4: 116, 5: 97, 6: 12, 7: 70, 8: 17, 9: 10, 10: 2,
        # 值从0开始取 所以它的长度可以想成unk的id映射
        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            for line in f.readlines():
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)  # 这里之所以加1  是因为第一维度是标签
                np_array[0] = int(values[1])
                for i in range(1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i+1], defaults[i])

                # struct把item_idx指定为字节流
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer
