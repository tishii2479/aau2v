import collections
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def attention(Q: Tensor, K: Tensor, V: Tensor):
    a = F.softmax(torch.matmul(Q, K.mT) / sqrt(K.size(2)), dim=2)
    return torch.matmul(a, V)


class UnigramSampler:
    def __init__(self, sequences, power):
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for sequence in sequences:
            for item in sequence:
                counts[item] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, batch_size: int, sample_size: int):
        # Ignores even if correct label is included
        negative_sample = np.random.choice(self.vocab_size, size=(batch_size, sample_size),
                                           replace=True, p=self.word_p)
        return negative_sample


class EmbeddingDot(nn.Module):
    def __init__(self, d_model: int, num_item: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_item, d_model)

    def forward(self, h: Tensor, indicies: Tensor):
        w = self.embedding.forward(indicies)
        w = torch.reshape(w, (-1, indicies.size(1), self.d_model))
        return torch.matmul(h, w.mT)


class NegativeSampling(nn.Module):
    def __init__(self, d_model: int, num_item: int, sequences, power: float = 0.75, sample_size: int = 5):
        super().__init__()
        self.d_model = d_model
        self.sample_size = sample_size
        self.sampler = UnigramSampler(sequences, power)
        self.embedding = EmbeddingDot(d_model, num_item)

    def forward(self, h, target_index):
        r'''
        Args:
            h size of (batch_size)
            target_index index, size of (batch_size)
        '''
        batch_size = target_index.size(0)

        # (batch_size, sample_size)
        negative_sample = torch.tensor(self.sampler.get_negative_sample(
            batch_size, self.sample_size), dtype=torch.long)

        h = torch.reshape(h, (batch_size, 1, self.d_model))

        # positive
        out = torch.sigmoid(self.embedding.forward(
            h, torch.reshape(target_index, (batch_size, 1))))
        label = torch.ones(batch_size, 1)
        out = torch.reshape(out, (batch_size, 1))
        positive_loss = F.binary_cross_entropy(out, label)

        # negative
        out = torch.sigmoid(self.embedding.forward(h, negative_sample))
        label = torch.zeros(batch_size, self.sample_size)
        out = torch.reshape(out, (batch_size, self.sample_size))
        negative_loss = F.binary_cross_entropy(out, label)

        loss = (positive_loss + negative_loss) / 2
        return loss


class Model(nn.Module):
    def __init__(
        self,
        num_seq: int,
        num_item: int,
        d_model: int,
        sequences,
        concat: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.concat = concat

        self.W_seq = nn.Embedding(num_seq, d_model)
        self.W_item = nn.Embedding(num_item, d_model)

        self.W_seq_key = nn.Linear(d_model, d_model)
        self.W_seq_value = nn.Linear(d_model, d_model)

        self.W_item_key = nn.Linear(d_model, d_model)
        self.W_item_value = nn.Linear(d_model, d_model)

        output_dim = d_model * 2 if concat else d_model
        self.output = NegativeSampling(output_dim, num_item, sequences)

    def forward(
        self,
        seq_index: Tensor,
        item_indicies: Tensor,
        target_index: Tensor
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

        if self.concat:
            c = torch.concat([c, self.W_seq_value(h_seq)], dim=1)
        else:
            c += self.W_seq_value(h_seq)
        loss = self.output.forward(c, target_index)
        return loss

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
        d_model: int,
        sequences
    ):
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

        out = F.softmax(v, dim=1)

        loss = F.cross_entropy(out, target_index)

        return loss

    @property
    def seq_embedding(self):
        return self.W_seq.weight

    @property
    def item_embedding(self):
        return self.W_item.weight
