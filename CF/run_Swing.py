"""
@file   : run_Swing.py
@time   : 2024-07-12
"""
import math
from itertools import combinations
import pandas as pd


alpha = 0.5
top_k = 20


def load_data(train_path):
    train_data = pd.read_csv(train_path, sep="\t", engine="python", names=["userid", "itemid", "rate"])  # 提取用户交互记录数据
    return train_data


def get_uitems_iusers(train):
    u_items = dict()
    i_users = dict()
    for index, row in train.iterrows():  # 处理用户交互记录
        u_items.setdefault(row["userid"], set())
        i_users.setdefault(row["itemid"], set())
        u_items[row["userid"]].add(row["itemid"])  # 得到user交互过的所有item
        i_users[row["itemid"]].add(row["userid"])  # 得到item交互过的所有user
    # print(u_items)   # {1: {101, 102, 103}, 2: {104, 101}, 3: {102, 103}, 4: {104, 101}, 5: {102, 103}}
    # print(i_users)    # {101: {1, 2, 4}, 102: {1, 3, 5}, 103: {1, 3, 5}, 104: {2, 4}}
    print("使用的用户个数为：{}".format(len(u_items)))   # 使用的用户个数为：5
    print("使用的item个数为：{}".format(len(i_users)))   # 使用的item个数为：4
    return u_items, i_users


def swing_model(u_items, i_users):
    # print(u_items)  # {1: {101, 102, 103}, 2: {104, 101}, 3: {102, 103}, 4: {104, 101}, 5: {102, 103}}
    # print(i_users)  # {101: {1, 2, 4}, 102: {1, 3, 5}, 103: {1, 3, 5}, 104: {2, 4}}

    item_pairs = list(combinations(i_users.keys(), 2))  # 全排列
    # print(item_pairs)  # [(101, 102), (101, 103), (101, 104), (102, 103), (102, 104), (103, 104)]
    print("item pairs length：{}".format(len(item_pairs)))

    item_sim_dict = dict()
    for (i, j) in item_pairs:
        # (101, 102)   i=101  j=102
        user_pairs = list(combinations(i_users[i] & i_users[j], 2))  # item_i和item_j对应的user取交集后全排列 得到user对
        # print(user_pairs)   # [(1, 3), (1, 5), (3, 5)]

        result = 0
        for (u, v) in user_pairs:
            result += (1 / math.sqrt(len(u_items[u]))) * (1 / math.sqrt(len(u_items[v]))) * 1 / (alpha + len(list(u_items[u] & u_items[v])))  # 分数公式

        if result != 0:
            item_sim_dict.setdefault(i, dict())   # {'101': {'102': score}}
            item_sim_dict[i][j] = format(result, '.6f')
    # print(item_sim_dict)  # {101: {104: '0.200000'}, 102: {103: '0.526599'}}
    return item_sim_dict


if __name__ == "__main__":
    top_k = 10  # 与item相似的前 k 个item

    train_df = pd.read_csv('ratings.txt', sep="\t", names=["userid", "itemid", "rate"])
    # print(train_df)
    """
            userid  itemid  rate
    0        1     101     5
    1        1     102     4
    2        1     103     3
    3        2     101     4
    4        2     104     5
    """

    u_items, i_users = get_uitems_iusers(train_df)
    # print(u_items)  # {1: {101, 102, 103}, 2: {104, 101}, 3: {102, 103}, 4: {104, 101}, 5: {102, 103}}
    # print(i_users)  # {101: {1, 2, 4}, 102: {1, 3, 5}, 103: {1, 3, 5}, 104: {2, 4}}

    item_sim_dict = swing_model(u_items, i_users)
    print(item_sim_dict)  # {101: {104: '0.200000'}, 102: {103: '0.526599'}}


