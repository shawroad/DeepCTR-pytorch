"""
# -*- coding: utf-8 -*-
# @File    : train.py
# @Time    : 2020/12/28 4:13 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F
from config import set_args
from data_process import read_data, feature_engine
from model import WideDeep


class MyDataSet(Dataset):
    def __init__(self, data):
        self.wide_data = data[0]
        self.deep_data = data[1]
        self.target = data[2]

    def __getitem__(self, index):
        wide_data = self.wide_data[index]
        deep_data = self.deep_data[index]
        target = self.target[index]
        data = (wide_data, deep_data, target)
        return data

    def __len__(self):
        return len(self.target)


def valid_epoch(model, valid_loader):
    model.eval()
    losses = []
    targets = []
    outs = []
    for idx, (data_wide, data_deep, target) in enumerate(valid_loader):
        data_wide, data_deep, target = data_wide.to(device), data_deep.to(device), target.to(device)
        x = (data_wide, data_deep)
        out = model(x)
        loss = F.binary_cross_entropy(out, target.float())
        losses.append(loss.item())
        targets += list(target.numpy())
        out = out.view(-1).detach().numpy()
        outs += list(np.int64(out > 0.5))
    acc = accuracy_score(targets, outs)
    return acc, sum(losses) / len(losses)


if __name__ == "__main__":
    args = set_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 加载数据
    data_path = './data/adult.data'
    data = read_data(data_path)
    train_data, test_data, deep_columns_idx, embedding_columns_dict = feature_engine(data)
    data_wide = train_data[0]
    train_data = (torch.from_numpy(train_data[0].values), torch.from_numpy(train_data[1].values),
                  torch.from_numpy(train_data[2].values))
    train_data = MyDataSet(train_data)

    test_data = (torch.from_numpy(test_data[0].values), torch.from_numpy(test_data[1].values),
                 torch.from_numpy(test_data[2].values))
    test_data = MyDataSet(test_data)
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    deep_model_params = {
        'deep_columns_idx': deep_columns_idx,
        'embedding_columns_dict': embedding_columns_dict,
        'hidden_size_list': args.hidden_size_list,
        'dropouts': args.dropouts,
        'deep_output_dim': args.deep_out_dim}
    wide_model_params = {
        'wide_input_dim': data_wide.shape[1],
        'wide_output_dim': args.wide_out_dim
    }
    widedeep = WideDeep(wide_model_params, deep_model_params)
    model = widedeep.to(device)
    optimizer = torch.optim.Adam(widedeep.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for idx, (data_wide, data_deep, target) in enumerate(trainloader):
            data_wide, data_deep, target = data_wide.to(device), data_deep.to(device), target.to(device)
            x = (data_wide, data_deep)
            optimizer.zero_grad()
            out = model(x)
            loss = F.binary_cross_entropy(out, target.float())
            print('epoch:{}, step:{}, loss:{:.10f}'.format(epoch, idx, loss))
            loss.backward()
            optimizer.step()
            if idx == len(trainloader):
                break

        acc, loss = valid_epoch(model, testloader)
        print('valid-- epoch:{}, loss:{}, acc:{}'.format(epoch, loss, acc))
        torch.save(model.state_dict(), "wide_deep_model_{}.pkl".format(epoch))

