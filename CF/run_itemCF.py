"""
@file   : run_itemCF.py
@time   : 2024-07-12
"""
import numpy as np
import pandas as pd


def loadData():
    items = {'A': {'Alice': 5.0, 'user1': 3.0, 'user2': 4.0, 'user3': 3.0, 'user4': 1.0},
             'B': {'Alice': 3.0, 'user1': 1.0, 'user2': 3.0, 'user3': 3.0, 'user4': 5.0},
             'C': {'Alice': 4.0, 'user1': 2.0, 'user2': 4.0, 'user3': 1.0, 'user4': 5.0},
             'D': {'Alice': 4.0, 'user1': 3.0, 'user2': 3.0, 'user3': 5.0, 'user4': 2.0},
             'E': {'user1': 3.0, 'user2': 5.0, 'user3': 4.0, 'user4': 1.0}
             }
    return items


if __name__ == '__main__':
    # 1. 加载数据
    item_data = loadData()

    # 2. 算两两商品的相似度
    similarity_matrix = pd.DataFrame(np.eye(len(item_data)), index=list(item_data.keys()), columns=list(item_data.keys()))
    # print(similarity_matrix)
    '''
         A    B    C    D    E
    A  1.0  0.0  0.0  0.0  0.0
    B  0.0  1.0  0.0  0.0  0.0
    C  0.0  0.0  1.0  0.0  0.0
    D  0.0  0.0  0.0  1.0  0.0
    E  0.0  0.0  0.0  0.0  1.0
    '''

    for i1, users1 in item_data.items():
        for i2, users2 in item_data.items():
            if i1 == i2:
                continue
            vec1, vec2 = [], []
            for user, rating1 in users1.items():
                rating2 = users2.get(user, -1)
                if rating2 == -1:
                    continue
                vec1.append(rating1)
                vec2.append(rating2)
            similarity_matrix[i1][i2] = np.corrcoef(vec1, vec2)[0][1]
    # print(similarity_matrix)
    """
              A         B         C         D         E
    A  1.000000 -0.476731 -0.123091  0.532181  0.969458
    B -0.476731  1.000000  0.645497 -0.310087 -0.478091
    C -0.123091  0.645497  1.000000 -0.720577 -0.427618
    D  0.532181 -0.310087 -0.720577  1.000000  0.581675
    E  0.969458 -0.478091 -0.427618  0.581675  1.000000
    """

    # 3. 推荐
    target_user = 'Alice'
    target_item = 'E'
    num = 2
    sim_items = []
    sim_items_list = similarity_matrix[target_item].sort_values(ascending=False).index.tolist()

    # 找出E中的相似商品  然后删除 目标用户没有做行为的商品
    for item in sim_items_list:  # top2
        # 如果target_user对物品item评分过
        if target_user in item_data[item]:
            sim_items.append(item)
        if len(sim_items) == num:
            break
    print(f'与物品{target_item}最相似的{num}个物品为：{sim_items}')

    # 4. 预测打分
    target_user_mean_rating = np.mean(list(item_data[target_item].values()))
    weighted_scores = 0.   # 分子
    corr_values_sum = 0.   # 分母
    target_item = 'E'
    for item in sim_items:
        corr_value = similarity_matrix[target_item][item]
        user_mean_rating = np.mean(list(item_data[item].values()))
        weighted_scores += corr_value * (item_data[item][target_user] - user_mean_rating)
        corr_values_sum += corr_value

    target_item_pred = target_user_mean_rating + weighted_scores / corr_values_sum
    print(f'用户{target_user}对物品{target_item}的预测评分为：{target_item_pred}')
