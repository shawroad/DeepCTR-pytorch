"""
@file   : run_userCF.py
@time   : 2024-07-12
"""
import numpy as np
import pandas as pd


def loadData():
    users = {'Alice': {'A': 5, 'B': 3, 'C': 4, 'D': 4},
             'user1': {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
             'user2': {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
             'user3': {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
             'user4': {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}
             }
    return users


if __name__ == '__main__':
    # 1. 加载数据
    user_data = loadData()

    # 2. 构建两两用户之间的相关性
    similarity_matrix = pd.DataFrame(np.eye(len(user_data)), index=list(user_data.keys()), columns=list(user_data.keys()))
    # print(similarity_matrix)
    '''
            Alice  user1  user2  user3  user4
    Alice    1.0    0.0    0.0    0.0    0.0
    user1    0.0    1.0    0.0    0.0    0.0
    user2    0.0    0.0    1.0    0.0    0.0
    user3    0.0    0.0    0.0    1.0    0.0
    user4    0.0    0.0    0.0    0.0    1.0
    '''

    for u1, items1 in user_data.items():
        # print(u1)   # Alice
        # print(items1)   # {'A': 5, 'B': 3, 'C': 4, 'D': 4}

        for u2, items2 in user_data.items():
            if u1 == u2:
                continue

            # 给每个用户搞一个向量
            vec1, vec2 = [], []
            for item, rating1 in items1.items():  # {'A': 5, 'B': 3, 'C': 4, 'D': 4}
                rating2 = items2.get(item, -1)
                if rating2 == -1:
                    continue
                vec1.append(rating1)
                vec2.append(rating2)
            # print(vec1)   # [5, 3, 4, 4]
            # print(vec2)   # [3, 1, 2, 3]
            # print(np.corrcoef(vec1, vec2))
            # 计算不同用户之间的皮尔逊相关系数
            similarity_matrix[u1][u2] = np.corrcoef(vec1, vec2)[0][1]
    print(similarity_matrix)

    # 3. 找出与目标用户最相似的用户
    target_user = 'Alice'
    num = 2  # top2
    # 由于最相似的用户为自己，去除本身
    sim_users = similarity_matrix[target_user].sort_values(ascending=False)[1:num+1].index.tolist()
    # print(sim_users)   # ['user1', 'user2']
    # print(f'与用户{target_user}最相似的{num}个用户为：{sim_users}')

    # 4. 基于皮尔逊相关系数预测用户评分
    target_item = 'E'
    weighted_scores = 0.   # 分子
    corr_values_sum = 0.   # 分母
    # 基于皮尔逊相关系数预测用户评分
    for user in sim_users:   # ['user1', 'user2']
        corr_value = similarity_matrix[target_user][user]   # 两个用户相似分
        user_mean_rating = np.mean(list(user_data[user].values()))  # 用户的平局打分

        weighted_scores += corr_value * (user_data[user][target_item] - user_mean_rating)  # 用户对该商品的打分减平均打分 乘 相似权重
        corr_values_sum += corr_value

    target_user_mean_rating = np.mean(list(user_data[target_user].values()))  # 目标用户的平均打分
    target_item_pred = target_user_mean_rating * weighted_scores / corr_values_sum
    print(f'用户{target_user}对物品{target_item}的预测评分为：{target_item_pred}')


