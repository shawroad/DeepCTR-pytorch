"""
@file   : data_helper.py
@time   : 2024-07-15
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



def load_data(train_path, test_path):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income_50k']

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    train_df.columns = column_names
    test_df.columns = column_names

    train_df['tag'] = 1
    test_df['tag'] = 0
    test_df['income_50k'] = test_df['income_50k'].apply(lambda x: x[:-1])   # 测试集有个多余的点
    df = pd.concat([train_df, test_df])
    df.dropna(inplace=True)

    label_columns = ['income_50k', 'marital_status']
    for col in label_columns:
        if col == 'income_50k':
            df[col] = df[col].apply(lambda x: 0 if x == ' <=50K' else 1)
        else:
            df[col] = df[col].apply(lambda x: 0 if x == ' Never-married' else 1)
    # print(df['income_50k'].value_counts())   #  0    37155   1    11687
    # print(df['marital_status'].value_counts())   # 1    32725   0    16117

    # 将类别特征转码
    categorical_columns = ['workclass', 'education', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    for col in column_names:
        if col not in label_columns + ['tag']:
            if col in categorical_columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            else:
                mm = MinMaxScaler()
                df[col] = mm.fit_transform(df[col].values.reshape(-1, 1))

    df = df[['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'occupation',
             'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
             'native_country', 'income_50k', 'marital_status', 'tag']]
    user_feature_dict, item_feature_dict = dict(), dict()
    for idx, col in enumerate(df.columns):
        if col not in label_columns + ['tag']:
            if idx < 7:
                if col in categorical_columns:
                    user_feature_dict[col] = (len(df[col].unique()) + 1, idx)
                else:
                    user_feature_dict[col] = (1, idx)
            else:
                if col in categorical_columns:
                    item_feature_dict[col] = (len(df[col].unique()) + 1, idx)
                else:
                    item_feature_dict[col] = (1, idx)

    train_df, test_df = df[df['tag'] == 1], df[df['tag'] == 0]

    train_df.drop('tag', axis=1, inplace=True)
    test_df.drop('tag', axis=1, inplace=True)
    return train_df, test_df, user_feature_dict, item_feature_dict


