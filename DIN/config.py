"""
@file   : config.py
@time   : 2024-07-12
"""
import argparse


def set_args():
    # 超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="./data/amazon-books-100k.txt", type=str)
    parser.add_argument('--output_dir', default='./output/', type=str, help="output")

    parser.add_argument('--epochs', type=int, default=50)

    # batch_size一次喂给模型多少个样本
    parser.add_argument('--batch_size', type=int, default=128, help="train batch size")

    parser.add_argument('--learning_rate', type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="weight_decay")
    parser.add_argument('--logging_steps', type=int, default=10, help="logging steps")
    parser.add_argument('--seed', type=int, default=42, help="seed")
    args = parser.parse_args()
    return args