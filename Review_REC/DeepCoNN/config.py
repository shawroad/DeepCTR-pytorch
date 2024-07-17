"""
@file   : config.py
@time   : 2024-07-16
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser(description='基于用户评论的推荐')
    parser.add_argument('--output_dir', type=str, default='./output', help='输入的模型保存的路径')
    parser.add_argument('--train_data', type=str, default='./data/train.csv', help='训练数据')
    parser.add_argument('--dev_data', type=str, default='./data/valid.csv', help='验证集')
    parser.add_argument('--word2vec_file', type=str, default='./data/glove.6B.50d.txt', help='词向量文件')

    parser.add_argument('--max_seq_len', type=int, default=128, help='最大长度')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练多少轮')

    parser.add_argument('--lowest_review_count', type=int, default=2)
    parser.add_argument('--review_length', type=int, default=40)
    parser.add_argument('--review_count', type=int, default=10)

    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--l2_regularization', type=float, default=1e-6, help='权重衰减程度')
    parser.add_argument('--learning_rate_decay', type=float, default=0.99, help='学习率衰减')

    parser.add_argument('--kernel_count', type=int, default=100, help='卷积核个数')

    parser.add_argument('--kernel_size', type=int, default=3, help='卷积核尺寸')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--cnn_out_dim', type=int, default=50, help='cnn的输出')

    parser.add_argument('--logging_steps', type=int, default=5, help='每间隔几步记录一次loss变化')
    parser.add_argument('--seed', default=2024, type=int, help='随机种子')

    args = parser.parse_args()
    return args