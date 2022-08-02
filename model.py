from torch import Tensor, nn
from math import sqrt
import torch
import torch.nn.functional as F


def attention(Q: Tensor, K: Tensor, V: Tensor):
    a = F.softmax(torch.matmul(Q, K.mT) / sqrt(K.size(2)), dim=2)
    return torch.matmul(a, V)


class Model(nn.Module):
    def __init__(
        self,
        num_seq: int,
        num_item: int,
        d_model: int
    ):
        super().__init__()
        self.d_model = d_model

        self.W_seq = nn.Embedding(num_seq, d_model)
        self.W_item = nn.Embedding(num_item, d_model)

        self.W_seq_key = nn.Linear(d_model, d_model)
        self.W_seq_value = nn.Linear(d_model, d_model)

        self.W_item_key = nn.Linear(d_model, d_model)
        self.W_item_value = nn.Linear(d_model, d_model)

        self.projection = nn.Linear(d_model * 2, num_item)

    def forward(
        self,
        seq_index: Tensor,
        item_indicies: Tensor
    ):
        r'''
        seq_index:
            type: `int` or `long`
            shape: (batch_size, )
        item_indicies:
            type: `int` or `long`
            shape: (batch_size, window_size)
        '''
        h_seq = self.W_seq.forward(seq_index)
        h_items = self.W_item.forward(item_indicies)

        Q = torch.reshape(self.W_seq_key(h_seq), (-1, 1, self.d_model))
        K = self.W_item_key(h_items)
        V = self.W_item_value(h_items)

        c = torch.reshape(attention(Q, K, V), (-1, self.d_model))
        c = torch.concat([c, self.W_seq_value(h_seq)], dim=1)

        v = self.projection.forward(c)

        return F.softmax(v, dim=1)

    @property
    def seq_embedding(self):
        return self.W_seq.weight

    @property
    def item_embedding(self):
        return self.W_item.weight


class MyDoc2Vec(nn.Module):
    def __init__(
        self,
        num_seq: int,
        num_item: int,
        d_model: int
    ):
        super().__init__()
        self.d_model = d_model

        self.W_seq = nn.Embedding(num_seq, d_model)
        self.W_item = nn.Embedding(num_item, d_model)

        self.projection = nn.Linear(d_model, num_item)

    def forward(
        self,
        seq_index: Tensor,
        item_indicies: Tensor
    ):
        r'''
        seq_index:
            type: `int` or `long`
            shape: (batch_size, )
        item_indicies:
            type: `int` or `long`
            shape: (batch_size, window_size)
        '''
        h_seq = self.W_seq.forward(seq_index)
        h_items = self.W_item.forward(item_indicies)

        h_seq = torch.reshape(h_seq, (-1, 1, self.d_model))
        c = torch.concat([h_seq] + [h_items], dim=1).mean(dim=1)

        v = self.projection.forward(c)

        return F.softmax(v, dim=1)

    @property
    def seq_embedding(self):
        return self.W_seq.weight

    @property
    def item_embedding(self):
        return self.W_item.weight
