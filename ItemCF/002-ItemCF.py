"""
# -*- coding: utf-8 -*-
# @File    : 002-ItemCF.py
# @Time    : 2020/12/15 1:57 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import math


def itemSimilarity(train):
    '''
    基于物品共现的方式
    :param train: data
    :return: 相似度矩阵
    '''
    C = dict()  # 书本对同时被购买的次数
    N = dict()  # 书本被购买用户数
    for user, items in train.items():
        # user用户  items物品 是个字典
        for i in items.keys():
            if i not in N.keys():
                N[i] = 0
            N[i] += 1    # 统计的是当前这个商品被多少人买过
            for j in items.keys():
                if i == j:
                    continue
                if i not in C.keys():
                    C[i] = dict()
                if j not in C[i].keys():
                    C[i][j] = 0
                C[i][j] += 1   # 这两个商品共现了  即被同一个人买了
    W = dict()
    for i, related_items in C.items():
        # i是某个物品  related_items是与i共现的物品 以及他们共现的次数
        if i not in W.keys():
            W[i] = dict()
        for j, cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])   # 同时购买这两中东西人的比例
    return W


def itemSimilarity_cos(train):
    '''
    基于余弦相似度算法
    :param train: data
    :return: 物品相似性矩阵
    '''
    C = dict()   # 书本被同时被购买的次数
    N = dict()   # 书本被购买用户数

    for user, items in train.items():
        for i in items.keys():
            if i not in N.keys():
                N[i] = 0
            N[i] += items[i] * items[i]
            for j in items.keys():
                if i == j:
                    continue
                if i not in C.keys():
                    C[i] = dict()
                if j not in C[i].keys():
                    C[i][j] = 0
                # 有用户同时购买了i, j   则加评分乘积
                C[i][j] += items[i] * items[j]

    W = dict()  # 书本对相似分数
    for i, related_items in C.items():
        if i not in W.keys():
            W[i] = dict()
        for j, cij in related_items.items():
            W[i][j] = cij / (math.sqrt(N[i]) * math.sqrt(N[j]))
    return W


def itemSimilarity_alpha(train, alpha=0.7):
    '''
    基于共现  并对热门商品进行惩罚
    :param train: data
    :param alpha: 惩罚因子
    :return: 物品相似性矩阵
    '''
    C = dict()   # 书本对同时被购买的次数
    N = dict()   # 书本被购买用户数
    for user, items in train.items():
        for i in items.keys():
            if i not in N.keys():
                N[i] = 0
            N[i] += 1
            for j in items.keys():
                if j == i:
                    continue
                if i not in C.keys():
                    C[i] = dict()
                if j not in C[i].keys():
                    C[i][j] = 0
                # 当用户同时购买了i和j 则加1
                C[i][j] += 1
    W = dict()  # 书本对相似分数
    for i, related_items in C.items():
        if i not in W.keys():
            W[i] = dict()
        for j, cij in related_items.items():
            W[i][j] = cij / (math.pow(N[i], alpha) * math.pow(N[j], 1 - alpha))
    return W


def recommend(train, user_id, W, K, top=5):
    '''
    开始推荐
    :param train: 数据
    :param user_id: 用户
    :param W: 商品的相似性矩阵
    :param K: 根据用户前几个购买商品进行推荐
    :param top: 推荐几个商品
    :return:
    '''
    rank = dict()
    ru = train[user_id]   # 用户喜欢的商品
    for i, pi in ru.items():
        tmp = W[i]   # 用户喜欢的某件商品
        for j, wj in sorted(tmp.items(), key=lambda d: d[1], reverse=True)[0:K]:
            # 对用户喜欢商品 进行的打分排序  然后找用户最喜欢商品的相似商品
            if j not in rank.keys():
                rank[j] = 0
            # 如果用户已经有此商品 则不再推荐
            if j in ru:
                continue
            # 待推荐的书本j与用户已购买的书本i相似，则累加上相似分数
            rank[j] += pi * wj   # 用户喜欢商品的打分 乘 当前待推荐商品的打分
    res = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:top]
    return res


if __name__ == '__main__':
    Train_Data = {
        'A': {'i1': 1, 'i2': 1, 'i4': 1},
        'B': {'i1': 1, 'i4': 1},
        'C': {'i1': 1, 'i2': 1, 'i5': 1},
        'D': {'i2': 1, 'i3': 1},
        'E': {'i3': 1, 'i5': 1},
        'F': {'i2': 1, 'i4': 1}
    }
    W = itemSimilarity_alpha(Train_Data)   # 基于惩罚的共现方式
    res = recommend(Train_Data, 'D', W, K=3, top=2)
    print(res)

