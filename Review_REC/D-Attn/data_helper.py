"""
@file   : data_helper.py
@time   : 2024-07-16
"""
import torch
import pandas as pd
from config import set_args
from torch.utils.data import Dataset


args = set_args()


def load_embedding(word2vec_file):
    with open(word2vec_file, encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_emb.append([0])
        word_dict['<UNK>'] = 0
        for line in f.readlines():
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])
            word_dict[tokens[0]] = len(word_dict)
        word_emb[0] = [0] * len(word_emb[1])
    return word_emb, word_dict


class Review_REDataset(Dataset):
    def __init__(self, data_path, word_dict, retain_rui=True):
        self.word_dict = word_dict
        self.PAD_WORD_idx = self.word_dict["<UNK>"]
        self.retain_rui = retain_rui  # 是否在最终样本中，保留user和item的公共review
        self.lowest_r_count = args.lowest_review_count  # lowest amount of reviews wrote by exactly one user/item
        self.review_length = args.review_length
        self.review_count = args.review_count

        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        df['review'] = df['review'].apply(self._review2id)  # 分词->数字
        # print(df.head())
        '''
            userID  itemID                                             review  rating
        0    3748     934  [366, 1780, 6381, 79575, 10268, 0, 1590, 17427...       4
        1    4795    2280  [3538, 1575, 9038, 1138, 0, 8391, 12971, 2685,...       5
        '''
        self.sparse_idx = set()  # 暂存稀疏样本的下标，最后删除他们
        user_reviews = self._get_reviews(df)  # 收集每个user的评论列表
        # print(user_reviews.size())   # torch.Size([51764, 10, 40])
        item_reviews = self._get_reviews(df, 'itemID', 'userID')
        # print(item_reviews.size())   # torch.Size([51764, 10, 40])

        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)

        self.user_reviews = user_reviews[[idx for idx in range(user_reviews.shape[0]) if idx not in self.sparse_idx]]
        self.item_reviews = item_reviews[[idx for idx in range(item_reviews.shape[0]) if idx not in self.sparse_idx]]
        self.rating = rating[[idx for idx in range(rating.shape[0]) if idx not in self.sparse_idx]]

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        # 对于每条训练数据，生成用户的所有评论汇总
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # 每个user/item评论汇总

        lead_reviews = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            # userid   itemid
            df_data = reviews_by_lead[lead_id]  # 取出lead的所有评论：DataFrame
            if self.retain_rui:
                reviews = df_data['review'].to_list()  # 取lead所有评论：列表
            else:
                reviews = df_data['review'][df_data[costar] != costar_id].to_list()  # 不含lead与costar的公共评论

            if len(reviews) < self.lowest_r_count:
                self.sparse_idx.add(idx)
            reviews = self._adjust_review_list(reviews, self.review_length, self.review_count)
            lead_reviews.append(reviews)
        return torch.LongTensor(lead_reviews)

    def _adjust_review_list(self, reviews, r_length, r_count):
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews))  # 评论数量固定
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]  # 每条评论定长
        return reviews

    def _review2id(self, review):
        # 将一个评论字符串分词并转为数字
        if not isinstance(review, str):
            return []
        wids = []
        for word in review.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])  # 单词映射为数字
            else:
                wids.append(self.PAD_WORD_idx)
        return wids