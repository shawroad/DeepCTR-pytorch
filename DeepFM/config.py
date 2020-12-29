"""
# -*- coding: utf-8 -*-
# @File    : config.py
# @Time    : 2020/12/28 7:02 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import argparse


def set_args():
    parse = argparse.ArgumentParser(description="wide deep model include arguments")
    # parse.add_argument("--hidden_size_list", nargs='+', type=int, default=[64, 32, 16])
    # parse.add_argument("--dropouts", nargs='+', type=int, default=[0.5, 0.5])
    # parse.add_argument("--deep_out_dim", default=1, type=int)
    # parse.add_argument("--wide_out_dim", default=1, type=int)
    # parse.add_argument("--batch_size", default=32, type=int)
    # parse.add_argument("--lr", default=0.01, type=float)
    # parse.add_argument("--print_step", default=200, type=int)
    parse.add_argument("--epochs", default=10, type=int)
    # parse.add_argument("--validation", default=True, type=bool)
    # parse.add_argument("--method", choices=['multiclass', 'binary', 'regression'], default='binary',type=str)
    args = parse.parse_args()
    return args