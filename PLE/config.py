"""
@file   : config.py
@time   : 2024-07-15
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="./data/adult.data", type=str)
    parser.add_argument("--test_data", default="./data/adult.test", type=str)
    parser.add_argument('--output_dir', default='./output/', type=str, help="output")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024, help="train batch size")

    parser.add_argument('--learning_rate', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="weight_decay")
    parser.add_argument('--n_gpu', type=int, default=0, help="n gpu")
    parser.add_argument('--seed', type=int, default=42, help="seed")
    args = parser.parse_args()
    return args