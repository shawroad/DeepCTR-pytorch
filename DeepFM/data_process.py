"""
# -*- coding: utf-8 -*-
# @File    : data_process.py
# @Time    : 2020/12/28 7:02 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import os
import random
import collections

# 有13个连续特征和26个离散特征
continous_features = range(1, 14)   # 之所以不从零开始  是因为0是标签数据
categorial_features = range(14, 40)


# 这里的特征是每个特征的最大取值  已经能容纳95%的数据，把超出这个阈值的特征 直接置为这个阈值
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


class CategoryDictGenerator:
    # 为每个离散特征生成一个字典  {特征1:对应的取值个数, 特征2:对应的取值个数, 特征3:对应的取值个数, ...}
    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=0):
        with open(datafile, 'r', encoding='utf8') as f:
            for line in f.readlines():
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
        # print(len(self.dicts))   # 26  每个元素是个字典， 统计的是当前特征的取值
        for i in range(0, self.num_feature):
            # 过滤低频的取值
            self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items())
            # 按出现的次数从大到小排序
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            # 加下来构建词表
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return [len(self.dicts[idx]) for idx in range(0, self.num_feature)]


class ContinuousFeatureGenerator:
    """
    Clip continuous features.
    """
    def __init__(self, num_feature):
        self.num_feature = num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r', encoding='utf8') as f:
            for line in f.readlines():
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        if val > continous_clip[i]:
                            val = continous_clip[i]

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return val


def preprocess(datadir, outdir):
    """
    对13个连续特征进行标准化，并把它们合并 组成一个13维的向量
    26个分类特征中的每一个都是一个热编码的，所有的一个one-hot向量被组合成一个稀疏的二进制向量。
    """
    # 1. 处理连续值
    dists = ContinuousFeatureGenerator(len(continous_features))
    dists.build(os.path.join(datadir, 'train.txt'), continous_features)

    # 2. 处理离散值
    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(os.path.join(datadir, 'train.txt'), categorial_features, cutoff=10)  # cutoff主要是为了过滤低频取值

    dict_sizes = dicts.dicts_sizes()
    # 每个离散特征的取值个数
    # print(dict_sizes)  # [10, 22, 7, 6, 7, 7, 3, 9, 3, 5, 6, 7, 7, 8, 5, 7, 10, 14, 4, 4, 7, 3, 9, 8, 11, 5]

    categorial_feature_offset = [0]
    for i in range(1, len(categorial_features)):
        offset = categorial_feature_offset[i-1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)
    # print(categorial_feature_offset)   # 偏移
    # [0, 10, 32, 39, 45, 52, 59, 62, 71, 74, 79, 85, 92, 99, 107, 112,119, 129, 143, 147, 151, 158, 161, 170, 178, 189]

    with open(os.path.join(outdir, 'feature_size.txt'), 'w') as f:
        sizes = [1] * len(continous_features) + dict_sizes
        sizes = [str(i) for i in sizes]
        # 前几个连续特征都是1  后面的离散特征取的是它取值的个数
        f.write(','.join(sizes))
    random.seed(0)

    # 保存数据
    # 训练集
    with open(os.path.join(outdir, 'train.txt'), 'w') as out_train:
        with open(os.path.join(datadir, 'train.txt'), 'r') as f:
            for line in f.readlines():
                features = line.rstrip('\n').split('\t')
                # 连续特征
                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i]])
                    continous_vals.append('{0:.6f}'.format(val).rstrip('0').rstrip('.'))
                # 离散特征
                categorial_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i]])
                    categorial_vals.append(str(val))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)
                label = features[0]
                out_train.write(','.join([continous_vals, categorial_vals, label]) + '\n')

    # 测试集
    with open(os.path.join(outdir, 'test.txt'), 'w') as out:
        with open(os.path.join(datadir, 'test.txt'), 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    continous_vals.append("{0:.6f}".format(val).rstrip('0').rstrip('.'))

                categorial_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i] - 1])
                    categorial_vals.append(str(val))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)
                out.write(','.join([continous_vals, categorial_vals]) + '\n')


if __name__ == "__main__":
    preprocess('./data/origin_data', './data')

