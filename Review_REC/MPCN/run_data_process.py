"""
@file   : run_data_process.py
@time   : 2024-07-16
"""
import argparse
import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


def load_vocab(path):
    all_vocab = []
    with open(path, 'r', encoding="utf8") as f:
        for line in f.readlines():
            line = line.strip()
            all_vocab.append(line)
    return all_vocab


def process_dataset(json_path, select_cols, train_rate, csv_path):
    df = pd.read_json(json_path, lines=True)
    df = df[select_cols]
    df.columns = ['userID', 'itemID', 'review', 'rating']
    df['userID'] = df.groupby(df['userID']).ngroup()   # ngroup:分配组号
    df['itemID'] = df.groupby(df['itemID']).ngroup()
    stop_words = load_vocab('./data/stopwords.txt')
    punctuations = load_vocab('./data/punctuations.txt')

    df = df.drop(df[[not isinstance(x, str) or len(x) == 0 for x in df['review']]].index)  # erase null reviews
    def clean_review(review):
        review = review.lower()
        for p in punctuations:
            review = review.replace(p, ' ')  # replace punctuations by space
        review = WordPunctTokenizer().tokenize(review)  # split words
        review = [word for word in review if word not in stop_words]  # remove stop words
        # review = [nltk.WordNetLemmatizer().lemmatize(word) for word in review]  # extract root of word
        return ' '.join(review)
    df['review'] = df['review'].apply(clean_review)
    train, valid = train_test_split(df, test_size=1 - train_rate, random_state=3)  # split dataset including random
    valid, test = train_test_split(valid, test_size=0.5, random_state=4)
    print(f'Split and saved dataset as csv: train {len(train)}, valid {len(valid)}, test {len(test)}')
    # Split and saved dataset as csv: train 51764, valid 6470, test 6471
    print(f'Total: {len(df)} reviews, {len(df.groupby("userID"))} users, {len(df.groupby("itemID"))} items.')
    # Total: 64705 reviews, 5541 users, 3568 items.
    return train, valid, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path',
                        default='./data/reviews_Digital_Music_5.json',
                        help='Selected columns of above dataset in json format.')
    parser.add_argument('--select_cols', dest='select_cols', nargs='+',
                        default=['reviewerID', 'asin', 'reviewText', 'overall'])
    # 'reviewerID', 'asin', 'reviewText', 'overall'
    # 'reviewerID' - 评论者ID  'asin' - 产品ID   'reviewText' - 评论内容  'overall' - 总体评分
    parser.add_argument('--train_rate', dest='train_rate', default=0.8)
    parser.add_argument('--save_dir', dest='save_dir', default='./music')
    args = parser.parse_args()
    train, valid, test = process_dataset(args.data_path, args.select_cols, args.train_rate, args.save_dir)
    train.to_csv('./data/train.csv', index=False, header=False)
    valid.to_csv('./data/valid.csv', index=False, header=False)
    test.to_csv('./data/test.csv', index=False, header=False)
