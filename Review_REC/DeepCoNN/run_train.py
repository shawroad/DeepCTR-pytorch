"""
@file   : run_train.py
@time   : 2024-07-16
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from config import set_args
from model import DeepCoNN
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_helper import load_embedding, Review_REDataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
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
        score = -val_loss
        # score = val_loss
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


def evaluate(model, data_loader):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            user_reviews, item_reviews, ratings = batch
            predict = model(user_reviews, item_reviews)
            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
        return mse / sample_count


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)

    word_emb, word_dict = load_embedding(args.word2vec_file)

    train_dataset = Review_REDataset(args.train_data, word_dict)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = Review_REDataset(args.dev_data, word_dict, retain_rui=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

    model = DeepCoNN(word_emb)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.l2_regularization)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.learning_rate_decay)

    tb_write = SummaryWriter()
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    early_stopping = EarlyStopping(patience=5, verbose=True)
    for epoch in range(args.num_epochs):
        model.train()
        total_loss, total_samples = 0, 0
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            user_reviews, item_reviews, ratings = batch
            # print(user_reviews.size())    # torch.Size([64, 10, 40])
            # print(item_reviews.size())    # torch.Size([64, 10, 40])
            # print(ratings.size())   # torch.Size([64, 1])

            predict = model(user_reviews, item_reviews)
            loss = F.mse_loss(predict, ratings, reduction='sum')  # 平方和误差
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch:{}, step:{}/{}, loss:{:.5f}".format(epoch, step, len(train_dataloader), loss.item()))

            total_loss += loss.item()
            total_samples += len(predict)

            global_step += 1
            tr_loss += loss.item()
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_write.add_scalar("train_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

        scheduler.step()
        model.eval()
        val_mse = evaluate(model, valid_dataloader)
        train_loss = total_loss / total_samples

        early_stopping(val_mse, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        log_str = "epoch: {}, train_loss:{}, val_mse:{}".format(epoch, train_loss, val_mse)
        logs_path = os.path.join(args.output_dir, 'logs.txt')
        with open(logs_path, 'a+') as f:
            s = log_str + '\n'
            f.write(s)

        tb_write.add_scalar("train_loss", train_loss, epoch)
        tb_write.add_scalar("val_mse", val_mse, epoch)
        model.train()





