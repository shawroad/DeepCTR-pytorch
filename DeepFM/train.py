"""
# -*- coding: utf-8 -*-
# @File    : train.py
# @Time    : 2020/12/28 7:02 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataloader import CriteoDataset
from model import DeepFM
from config import set_args


def evaluate(eval_iter, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for xi, xv, y in eval_iter:
            xi = xi.to(device=device)
            xv = xv.to(device=device)
            y = y.to(device=device)
            output = model(xi, xv)
            preds = (F.sigmoid(output) > 0.5).to(torch.long)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('准确率:{:.10f}'.format(acc))
        return acc


if __name__ == '__main__':
    args = set_args()
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    # 加载数据
    train_data = CriteoDataset('data', train=True)
    train_iter = DataLoader(train_data, batch_size=32, shuffle=True)

    eval_data = CriteoDataset('data', train=True)
    eval_iter = DataLoader(eval_data, batch_size=32, shuffle=True)

    feature_sizes = np.loadtxt('./data/feature_size.txt', delimiter=',')
    feature_sizes = [int(x) for x in feature_sizes]
    # print(feature_sizes)
    model = DeepFM(feature_sizes)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)

    model.to(device)
    # criterion = F.binary_cross_entropy_with_logits
    for epoch in range(args.epochs):
        for step, (xi, xv, y) in enumerate(train_iter):
            xi, xv, y = xi.to(device, dtype=torch.long), xv.to(device), y.to(device)
            output = model(xi, xv)
            # print(total.size())   # torch.Size([32])
            loss = F.binary_cross_entropy(output, y.float())
            print('epoch:{}, step: {}, loss:{}'.format(epoch, step, loss))
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()

        acc = evaluate(eval_iter, model)
        print('eval_data: epoch:{}, accuracy:{}'.format(epoch, acc))
        # 保存模型
        torch.save(model.state_dict(), "DeepFM_model_{}.pkl".format(epoch))


