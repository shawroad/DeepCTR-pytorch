"""
@file   : run_train.py
@time   : 2024-07-12
"""
import os
import torch
import pandas as pd
from torch import nn
import numpy as np
from tqdm import tqdm
from torch import optim
from config import set_args
from model import DeepInterestNet
from data_helper import process_data
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter


# 早停机制
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(args.output_dir, path)
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        # score = -val_loss
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def evaluate_model(model):
    model.eval()
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        val_loss = 0
        for step, x in tqdm(enumerate(valid_loader)):
            feat, label = x[0], x[1]
            if torch.cuda.is_available():
                feat, label = feat.cuda(), label.cuda()
            pred = model(feat)
            pred = pred.view(-1)
            loss = loss_func(pred, label)
            val_loss += loss.item()
            logits = pred.data.cpu().numpy().tolist()
            valid_preds.extend(logits)
            valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        val_loss = val_loss / len(valid_loader)
        return cur_auc, val_loss


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # 读取数据集
    df = pd.read_csv(args.train_data)

    # print(df.head())
    # print(df.shape)  # (89999, 6)
    # 数据处理
    df = process_data(df)
    # print(df.head())

    # hist_cate_0  hist_cate_1  hist_cate_2  ...  hist_cate_39  cateID  label
    # 用户的四十个兴趣的类别   cateID: 推荐的商品的类别

    fields = df.max().max()   # 统计出所有商品类别的最大编号

    x_data = df.drop(['label'], axis=1)
    y_data = df['label']
    # print(x_data)
    # print(y_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=1, stratify=y_data)
    # print(x_train.shape)   # (71999, 41)
    # print(x_test.shape)   # (18000, 41)

    train_dataset = TensorDataset(torch.LongTensor(x_train.values), torch.FloatTensor(y_train.values))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = TensorDataset(torch.LongTensor(x_test.values), torch.FloatTensor(y_test.values))
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)

    # 模型部分
    model = DeepInterestNet(feature_dim=fields, embed_dim=8, mlp_dims=[64, 32], dropout=0.2)
    if torch.cuda.is_available():
        model.cuda()

    # 交叉熵损失
    loss_func = nn.BCELoss()   # 交叉熵损失   # focal loss

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # 权重的衰减

    # 学习率的调整
    # 线性衰减    cos衰减   模型的预热
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.batch_size, gamma=0.8)

    best_auc = 0.0
    early_stopping = EarlyStopping(patience=10, verbose=True)
    tb_write = SummaryWriter()
    global_step, tr_loss, logging_loss = 0, 0, 0
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        for step, x in enumerate(train_loader):
            feat, label = x[0], x[1]
            # print(feat.shape)   # torch.Size([128, 41])
            # print(label.shape)   # torch.Size([128])

            if torch.cuda.is_available():
                feat, label = feat.cuda(), label.cuda()

            pred = model(feat)
            pred = pred.view(-1)
            loss = loss_func(pred, label)  # 算损失

            optimizer.zero_grad()
            loss.backward()  # 反向计算梯度
            optimizer.step()  # 更新梯度
            train_loss_sum += loss.item()
            print("Epoch {:04d} | step {:04d} / {} | loss {:.4f}".format(
                epoch + 1, step + 1, len(train_loader), train_loss_sum / (step + 1)))

            tr_loss += loss.item()
            global_step += 1
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_write.add_scalar("train_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

        scheduler.step()
        cur_auc, val_loss = evaluate_model(model)

        tb_write.add_scalar("val_auc", cur_auc, epoch)
        tb_write.add_scalar("val_loss", val_loss, epoch)

        log_path = os.path.join(args.output_dir, 'logs.txt')
        train_loss = train_loss_sum / len(train_loader)

        with open(log_path, 'a+') as f:
            ss = 'epoch:{}, train_loss:{}, val_loss: {}, val_auc:{}'.format(epoch + 1, train_loss, val_loss, cur_auc)
            f.write(ss + '\n')

        early_stopping(cur_auc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.bin'))
























