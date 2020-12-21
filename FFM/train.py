"""
# -*- coding: utf-8 -*-
# @File    : train.py
# @Time    : 2020/12/18 4:03 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from dataloader import AvazuDataset
from model import FieldAwareFactorizationMachineModel
from config import set_args


class EarlyStopper:
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


if __name__ == '__main__':
    args = set_args()
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    # 1. 加载数据
    dataset = AvazuDataset(args.train_data_path)
    # 2. 切分训练集 验证集 测试集
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=2)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2)

    # print(dataset.field_dims)   # 也就是每个特征取值的个数加1
    # [  5   6   3 117  98  13  71  18  11   3  63 257   5   5 197   4   5 123
    #    5  37  63  30]
    model = FieldAwareFactorizationMachineModel(dataset.field_dims, embed_dim=4)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{args.save_dir}/{args.model_name}.pt')
    for epoch in range(args.Epoch):
        model.train()
        total_loss = 0
        for step, (fields, target) in enumerate(train_data_loader):
            fields, target = fields.to(device), target.to(device)
            # print(fields.size())   # torch.Size([32, 22])   batch_size x 特征值个数
            # print(target.size())   # torch.Size([32])   batch_size
            y = model(fields)
            loss = criterion(y, target.float())
            print('epoch: {}, step: {}, loss: {}'.format(epoch, step, loss))
            model.zero_grad()
            loss.backward()
            optimizer.step()

        auc = test(model, valid_data_loader, device)
        print('dev epoch:{}, acc:{}'.format(epoch, auc))
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')

