"""
# -*- coding: utf-8 -*-
# @File    : config.py
# @Time    : 2020/12/17 2:48 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import argparse

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--train_data_path', type=str, default='../data/train')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--Epoch', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./save_model')
    parser.add_argument('--model_name', type=str, default='fm')
    args = parser.parse_args()
    return args

