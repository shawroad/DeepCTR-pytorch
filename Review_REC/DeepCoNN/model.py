"""
@file   : model.py
@time   : 2024-07-16
"""
import torch
from torch import nn
from config import set_args

args = set_args()


class CNN(nn.Module):
    def __init__(self, word_dim):
        super(CNN, self).__init__()
        self.kernel_count = args.kernel_count
        self.review_count = args.review_count
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=word_dim,
                out_channels=args.kernel_count,
                kernel_size=args.kernel_size,
                padding=(args.kernel_size - 1) // 2),  # out shape(new_batch_size, kernel_count, review_length)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, args.review_length)),  # out shape(new_batch_size,kernel_count,1)
            nn.Dropout(p=args.dropout_prob)
        )

        self.linear = nn.Sequential(
            nn.Linear(args.kernel_count * args.review_count, args.cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(p=args.dropout_prob))

    def forward(self, vec):
        # print(vec.size())   # torch.Size([640, 40, 50])
        latent = self.conv(vec.permute(0, 2, 1))  # out(new_batch_size, kernel_count, 1) kernel count指一条评论潜在向量
        # print(latent.size())   # torch.Size([640, 100, 1])
        latent = self.linear(latent.reshape(-1, self.kernel_count * self.review_count))
        return latent


class FactorizationMachine(nn.Module):
    def __init__(self, p, k):  # p=cnn_out_dim
        super().__init__()
        self.v = nn.Parameter(torch.rand(p, k) / 10)
        self.linear = nn.Linear(p, 1, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        linear_part = self.linear(x)
        # print(linear_part.size())   # torch.Size([64, 1])

        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)
        pair_interactions = self.dropout(pair_interactions)
        output = linear_part + 0.5 * pair_interactions
        return output  # out shape(batch_size, 1)


class DeepCoNN(nn.Module):
    def __init__(self, word_emb):
        super(DeepCoNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.cnn_u = CNN(word_dim=self.embedding.embedding_dim)
        self.cnn_i = CNN(word_dim=self.embedding.embedding_dim)
        self.fm = FactorizationMachine(args.cnn_out_dim * 2, 10)

    def forward(self, user_review, item_review):
        # print(user_review.size())   # torch.Size([64, 10, 40])
        # print(item_review.size())   # torch.Size([64, 10, 40])
        new_batch_size = user_review.shape[0] * user_review.shape[1]
        user_review = user_review.reshape(new_batch_size, -1)
        item_review = item_review.reshape(new_batch_size, -1)
        # print(user_review.size())   # torch.Size([640, 40])
        # print(item_review.size())   # torch.Size([640, 40])

        u_vec = self.embedding(user_review)
        i_vec = self.embedding(item_review)
        # print(u_vec.size())   # torch.Size([640, 40, 50])
        # print(i_vec.size())   # torch.Size([640, 40, 50])

        user_latent = self.cnn_u(u_vec)
        # print(user_latent.size())    # torch.Size([64, 50])
        item_latent = self.cnn_i(i_vec)
        # print(item_latent.size())    # torch.Size([64, 50])

        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        prediction = self.fm(concat_latent)
        # print(prediction.size())   # torch.Size([64, 1])
        return prediction