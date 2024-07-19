"""
@file   : run_train.py
@time   : 2024-07-15
#
"""
import os
import torch
import numpy as np
from torch import nn
from torch import optim
from model import PLE
from tqdm import tqdm
from config import set_args
from data_helper import load_data
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')



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


def evaluate_model(model, valid_loader):
    model.eval()
    with torch.no_grad():
        valid_labels1, valid_preds1 = [], []
        valid_labels2, valid_preds2 = [], []
        val_loss = 0
        for step, x in tqdm(enumerate(valid_loader)):
            feat, y1, y2 = x[0], x[1], x[2]
            if torch.cuda.is_available():
                feat, y1, y2 = feat.cuda(), y1.cuda(), y2.cuda()

            predict = model(feat)
            loss_1 = loss_func(predict[0], y1)  #
            loss_2 = loss_func(predict[1], y2)  #
            loss = loss_1 + loss_2

            val_loss += loss.item()
            logits1 = predict[0].squeeze().data.cpu().numpy().tolist()
            logits2 = predict[1].squeeze().data.cpu().numpy().tolist()
            valid_labels1.extend(y1.cpu().numpy().tolist())
            valid_labels2.extend(y2.cpu().numpy().tolist())
            valid_preds1.extend(logits1)
            valid_preds2.extend(logits2)

        auc1 = roc_auc_score(valid_labels1, valid_preds1)
        auc2 = roc_auc_score(valid_labels2, valid_preds2)
        auc = (auc1 + auc2) / 2
        val_loss = val_loss / len(valid_loader)
        return auc, val_loss, auc1, auc2


if __name__ == "__main__":
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_df, test_df, user_feature_dict, item_feature_dict = load_data(args.train_data, args.test_data)
    # print(user_feature_dict)
    # {'age': (1, 0), 'workclass': (10, 1), 'fnlwgt': (1, 2), 'education': (17, 3),
    # 'education_num': (1, 4), 'occupation': (16, 5), 'relationship': (7, 6)}
    # print(item_feature_dict)
    # {'race': (6, 7), 'sex': (3, 8), 'capital_gain': (1, 9), 'capital_loss': (1, 10),
    # 'hours_per_week': (1, 11), 'native_country': (43, 12)}

    train_dataset = TensorDataset(torch.LongTensor(train_df.iloc[:, :-2].values),
                                  torch.FloatTensor(train_df.iloc[:, -2:-1].values),
                                  torch.FloatTensor(train_df.iloc[:, -1:].values))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = TensorDataset(torch.LongTensor(test_df.iloc[:, :-2].values),
                                  torch.FloatTensor(test_df.iloc[:, -2:-1].values),
                                  torch.FloatTensor(test_df.iloc[:, -1:].values))
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = PLE(user_feature_dict, item_feature_dict, emb_dim=64)
    if torch.cuda.is_available():
        model.cuda()

    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.batch_size, gamma=0.8)

    best_auc = 0.0
    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        for step, x in enumerate(train_loader):
            feat, y1, y2 = x[0], x[1], x[2]
            # print(feat.size())   # torch.Size([16, 13])
            # print(y1.size())   # torch.Size([16])
            # print(y2.size())   # torch.Size([16])
            if torch.cuda.is_available():
                feat, y1, y2 = feat.cuda(), y1.cuda(), y2.cuda()

            predict = model(feat)
            # print(predict[0].size())   # torch.Size([16, 1])
            # print(predict[1].size())   # torch.Size([16, 1])
            loss_1 = loss_func(predict[0], y1)   #
            loss_2 = loss_func(predict[1], y2)   #
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            print("Epoch {:04d} | step {:04d} / {} | loss1 {:.4f}, loss2 {:.4f}, total_loss:{:.4f}".format(
                epoch + 1, step + 1, len(train_loader), loss_1.item(), loss_2.item(), loss.item()))

        model.eval()
        scheduler.step()
        auc, val_loss, auc1, auc2 = evaluate_model(model, valid_loader)
        log_path = os.path.join(args.output_dir, 'logs.txt')
        train_loss = train_loss_sum / len(train_loader)
        with open(log_path, 'a+') as f:
            ss = 'epoch:{}, train_loss:{}, val_loss: {}, val_auc:{}, val_auc1:{}, val_auc2:{}'.format(epoch + 1, train_loss, val_loss, auc, auc1, auc2)
            f.write(ss + '\n')

        early_stopping(auc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # if cur_auc > best_auc:
        #     best_auc = cur_auc
        #     output_model_file = os.path.join(args.output_dir, "best_model_{}.bin".format(epoch))
        #     torch.save(model.state_dict(), output_model_file)
