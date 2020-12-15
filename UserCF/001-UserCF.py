"""
# -*- coding: utf-8 -*-
# @File    : 001-UserCF.py
# @Time    : 2020/12/15 2:29 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import math


def build_item_index(data):
    '''
    建立物品倒排表
    :param data:
    :return: {物品: {人1: 评分, 人2: 评分...}}
    '''
    dictItem = dict()
    for user in data.keys():
        for item, score in data[user].items():
            if item not in dictItem.keys():
                dictItem[item] = dict()
            dictItem[item][user] = score
    return dictItem


def userSimilarity(dictItem):
    '''
    :param dictItem: 物品的倒排  eg:{物品: {人1: 评分, 人2: 评分...}}
    :return: 用户的相似度矩阵
    '''
    N = dict()   # 用户购买的数量
    C = dict()
    W = dict()
    for item in dictItem.keys():
        for user in dictItem[item].keys():
            if user not in N.keys():
                N[user] = 0
            N[user] += 1   # 用户购买的数量

            for user2 in dictItem[item].keys():
                if user2 == user:
                    continue   # 是同一个用户
                if user not in C.keys():
                    C[user] = dict()
                if user2 not in C[user].keys():
                    C[user][user2] = 0
                C[user][user2] += 1

    for i, related_user in C.items():
        if i not in W.keys():
            W[i] = dict()
        for j, cij in related_user.items():
            if j not in W[i].keys():
                W[i][j] = 0
            W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W


def recommend(train, user_id, W, K, top=5):
    rank = dict()
    rela_user = W[user_id]  # 当前用户相关的一些用户

    # 遍历最相关的三个用户
    for user, wi in sorted(rela_user.items(), key=lambda d:d[1], reverse=True)[:K]:
        # 然后看这几个用户喜欢的产品
        for item, wj in train[user].items():
            if item in train[user_id].keys():
                continue
            else:
                rank[item] = wi * wj
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
    dictItem = build_item_index(Train_Data)
    W = userSimilarity(dictItem)
    # for user, rela_user in W.items():
    #     print('{}: {}'.format(user, rela_user))

    user_id = 'B'   # 给用户推荐商品
    K = 3   # 看与他最相关的三个用户的喜欢产品
    top = 5  # 推荐几个产品给他
    res = recommend(Train_Data, user_id, W, K, top=5)
    print(res)
