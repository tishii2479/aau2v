from math import sqrt
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from layer import NegativeSampling


class AttentiveModel(nn.Module):
    def __init__(
        self,
        num_seq: int,
        num_item: int,
        d_model: int,
        sequences: List[List[int]],
        concat: bool = False,
        negative_sample_size: int = 30
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.concat = concat

        self.W_seq = nn.Embedding(num_seq, d_model)
        self.W_item = nn.Embedding(num_item, d_model)

        self.W_item_key = nn.Linear(d_model, d_model)
        self.W_item_value = nn.Linear(d_model, d_model)

        output_dim = d_model * 2 if concat else d_model
        self.output = NegativeSampling(
            d_model=output_dim, num_item=num_item,
            sequences=sequences, negative_sample_size=negative_sample_size)

    def forward(
        self,
        seq_index: Tensor,
        item_indicies: Tensor,
        target_index: Tensor
    ) -> Tensor:
        r'''
        seq_index:
            type: `int` or `long`
            shape: (batch_size, )
        item_indicies:
            type: `int` or `long`
            shape: (batch_size, window_size)
        target_index:
            type: `int` or `log`
            shape: (batch_size, 1 or sample_size)
        '''
        def attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
            a = F.softmax(torch.matmul(Q, K.mT) / sqrt(K.size(2)), dim=2)
            return torch.matmul(a, V)

        h_seq = self.W_seq.forward(seq_index)
        h_items = self.W_item.forward(item_indicies)

        Q = torch.reshape(h_seq, (-1, 1, self.d_model))
        K = self.W_item_key(h_items)
        V = self.W_item_value(h_items)

        c = torch.reshape(attention(Q, K, V), (-1, self.d_model))

        if self.concat:
            c = torch.concat([c, h_seq], dim=1)
        else:
            c += h_seq
        loss = self.output.forward(c, target_index)
        return loss

    @property
    def seq_embedding(self) -> Tensor:
        return self.W_seq.weight.data

    @property
    def item_embedding(self) -> Tensor:
        return self.W_item.weight.data


class OriginalDoc2Vec(nn.Module):
    def __init__(
        self,
        num_seq: int,
        num_item: int,
        d_model: int,
        sequences: List[List[int]]
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.W_seq = nn.Embedding(num_seq, d_model)
        self.W_item = nn.Embedding(num_item, d_model)

        self.projection = nn.Linear(d_model, num_item)

    def forward(
        self,
        seq_index: Tensor,
        item_indicies: Tensor,
        target_index: Tensor
    ) -> Tensor:
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

        out = F.softmax(v, dim=1)

        loss = F.cross_entropy(out, target_index)

        return loss

    @property
    def seq_embedding(self) -> Tensor:
        return self.W_seq.weight

    @property
    def item_embedding(self) -> Tensor:
        return self.W_item.weight
