import collections
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class UnigramSampler:
    def __init__(self, sequences: List[List[int]], power: float) -> None:
        counts: collections.Counter = collections.Counter()
        for sequence in sequences:
            for item in sequence:
                counts[item] += 1

        self.vocab_size = len(counts)

        self.word_p = np.zeros(self.vocab_size)
        for i in range(self.vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(
        self, batch_size: int, negative_sample_size: int
    ) -> np.ndarray:
        # Ignores even if correct label is included
        negative_sample = np.random.choice(self.vocab_size, size=(
            batch_size, negative_sample_size), replace=True, p=self.word_p)
        return negative_sample


class EmbeddingDot(nn.Module):
    def __init__(self, d_model: int, num_item: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_item, d_model)

    def forward(self, h: Tensor, indicies: Tensor) -> Tensor:
        w = self.embedding.forward(indicies)
        w = torch.reshape(w, (-1, indicies.size(1), self.d_model))
        out = torch.matmul(h, w.mT)
        return out


class NegativeSampling(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_item: int,
        sequences: List[List[int]],
        power: float = 0.75,
        negative_sample_size: int = 5
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.negative_sample_size = negative_sample_size
        self.sampler = UnigramSampler(sequences, power)
        self.embedding = EmbeddingDot(d_model, num_item)

    def forward(self, h: Tensor, target_index: Tensor) -> Tensor:
        r'''
        Args:
            h size of (batch_size)
            target_index index, size of (batch_size)
        '''
        batch_size = target_index.size(0)

        h = torch.reshape(h, (batch_size, 1, self.d_model))

        # positive
        out = torch.sigmoid(self.embedding.forward(
            h, torch.reshape(target_index, (batch_size, 1))))
        label = torch.ones(batch_size, 1)
        out = torch.reshape(out, (batch_size, 1))
        positive_loss = F.binary_cross_entropy(out, label)

        # negative
        # (batch_size, negative_sample_size)
        negative_sample = torch.tensor(self.sampler.get_negative_sample(
            batch_size, self.negative_sample_size), dtype=torch.long)

        out = torch.sigmoid(self.embedding.forward(h, negative_sample))
        label = torch.zeros(batch_size, self.negative_sample_size)
        out = torch.reshape(out, (batch_size, self.negative_sample_size))
        negative_loss = F.binary_cross_entropy(out, label)

        loss = (positive_loss + negative_loss) / 2
        return loss
