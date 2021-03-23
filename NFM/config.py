"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-23
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data/criteo_sampled_data.csv", type=str)
    parser.add_argument('--Epochs', type=int, default=10)

    parser.add_argument('--train_batch_size', type=int, default=2, help="train batch size")
    parser.add_argument('--eval_batch_size', type=int, default=2, help="eval batch size")

    parser.add_argument('--learning_rate', type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="weight_decay")
    parser.add_argument('--n_gpu', type=int, default=0, help="n gpu")
    args = parser.parse_args()
    return args