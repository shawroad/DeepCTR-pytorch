"""
# -*- coding: utf-8 -*-
# @File    : data_process.py
# @Time    : 2020/12/28 4:13 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split


def read_data(data_path):
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
             'native-country', 'target']
    data = pd.read_table(data_path, sep=',', names=names)
    return data


def feature_engine(data):
    # 标签
    data['target'] = data['target'].apply(lambda x: 0 if x.strip() == '<=50K' else 1)

    # age
    bins = [-np.inf, 18, 25, 35, 45, 50, np.inf]
    labels = list(range(len(bins) - 1))
    data['age'] = pd.cut(data['age'], bins=bins, labels=labels)   # 对年龄的连续值离散化

    # education-num
    bins = [-np.inf, 5, 10, 20, 40, np.inf]
    labels = list(range(len(bins) - 1))
    data['education-num'] = pd.cut(data['education-num'], bins=bins, labels=labels)

    # hours-per-week
    bins = [-np.inf, 10, 30, 40, 70, np.inf]
    labels = list(range(len(bins) - 1))
    data['hours-per-week'] = pd.cut(data['hours-per-week'], bins=bins, labels=labels)

    # 除去连续值和我们之前处理过的值  然后用labelEncoder进行编码
    continuous_cols = ['fnlwgt', 'capital-gain', 'capital-loss']
    cat_columns = [col for col in data.columns if col not in continuous_cols + ['age', 'hours-per-week', 'education-num']]

    for col in cat_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # 针对连续值进行标准化
    for col in continuous_cols:
        mms = MinMaxScaler()
        data[col] = mms.fit_transform(data[col].values.reshape(-1, 1)).reshape(-1)
    # 此时 把连续值能离散化的已经离散化了，该归一化的已经归一化了。
    # print(data.shape)   # (32561, 15)

    # 自己做的交叉特征
    wide_columns = ['age', 'workclass', 'education', 'education-num', 'occupation', 'relationship',
                    'hours-per-week', 'native-country', 'marital-status', 'sex']
    data_wide = data[wide_columns]
    cross_columns = [['occupation', 'sex'], ['occupation', 'education'], ['education', 'native-country'],
                     ['age', 'occupation'], ['age', 'hours-per-week'], ['sex', 'education']]
    for l in cross_columns:
        poly = PolynomialFeatures()
        c = poly.fit_transform(data_wide[l])    # 取出待交叉的两个特征 进行多项式表达  然后多项式的权重为特征。 因为默认都是5个权重
        c = pd.DataFrame(c, columns=[l[0] + '_' + l[1] + '_{}'.format(i) for i in range(c.shape[1])])
        data_wide = pd.concat((data_wide, c), axis=1)
    # print(data_wide.shape)   # (32561, 46)  # 单一特征+交叉特征  总共46个

    # onehot  之前的非交叉特征  将其作为one-hot
    for col in wide_columns:
        data_wide[col] = data_wide[col].astype('str')
    data_wide = pd.get_dummies(data_wide)    # 离散值做one-hot
    data_target = data['target']
    # 上面的data_wide 我们认为是输入到wide模型中的特征

    # 构建embedding dict
    deep_columns = ['workclass', 'occupation', 'native-country', 'race', 'fnlwgt', 'capital-gain', 'capital-loss']
    data_deep = data[deep_columns]
    embedding_columns = ['workclass', 'occupation', 'native-country', 'race']
    embedding_columns_dict = {}
    for i in range(len(deep_columns)):
        if deep_columns[i] in embedding_columns:
            col_name = deep_columns[i]
            embedding_columns_dict[col_name] = (len(data_deep[col_name].unique()), 8)
    deep_columns_idx = dict()
    for idx, key in enumerate(data_deep.columns):
        deep_columns_idx[key] = idx

    # print(embedding_columns_dict)
    # {'workclass': (9, 8), 'occupation': (15, 8), 'native-country': (42, 8), 'race': (5, 8)}
    # print(deep_columns_idx)
    # {'workclass': 0, 'occupation': 1, 'native-country': 2, 'race': 3, 'fnlwgt': 4, 'capital-gain': 5, 'capital-loss': 6}

    train_wide, test_wide = train_test_split(data_wide, test_size=0.4, random_state=999)
    train_deep, test_deep = train_test_split(data_deep, test_size=0.4, random_state=999)
    train_target, test_target = train_test_split(data_target, test_size=0.4, random_state=999)
    train, test = (train_wide, train_deep, train_target), (test_wide, test_deep, test_target)
    return train, test, deep_columns_idx, embedding_columns_dict


if __name__ == "__main__":
    # 1. 加载数据
    data_path = './data/adult.data'
    data = read_data(data_path)
    train, test, deep_columns_idx, embedding_columns_dict = feature_engine(data)
