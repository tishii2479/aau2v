import collections
import math
from math import sqrt
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
    return F.cosine_similarity(a, b)


def attention_weight(Q: Tensor, K: Tensor) -> Tensor:
    dim = len(Q.shape) - 1  # to handle batched and unbatched data
    return F.softmax(torch.matmul(Q, K.mT) / sqrt(K.size(dim)), dim=dim)


def attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    a = attention_weight(Q, K)
    return torch.matmul(a, V)


def calc_weighted_meta(h_meta: Tensor, meta_weights: Tensor) -> Tensor:
    return torch.matmul(
        h_meta.mT,
        meta_weights.view((*h_meta.shape[:-1], 1)),
    ).squeeze()


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
        negative_sample = np.random.choice(
            self.vocab_size,
            size=(batch_size, negative_sample_size),
            replace=True,
            p=self.word_p,
        )
        return negative_sample


class EmbeddingDot(nn.Module):
    def __init__(
        self, d_model: int, num_item: int, max_embedding_norm: Optional[float] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = MyEmbedding(num_item, d_model, max_norm=max_embedding_norm)

    def forward(self, h: Tensor, indicies: Tensor) -> Tensor:
        """Forward Embedding Dot

        Args:
            h (Tensor): input, size of (batch_size, 1, d_model)
            indicies (Tensor): indicies selected to get embedding,
                size of (batch_size, 1 or sample_size)

        Returns:
            Tensor: output
        """
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
        negative_sample_size: int = 5,
        max_embedding_norm: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.negative_sample_size = negative_sample_size
        self.sampler = UnigramSampler(sequences, power)
        self.embedding = EmbeddingDot(
            d_model, num_item, max_embedding_norm=max_embedding_norm
        )

    def forward(
        self, h: Tensor, target_index: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            h size of (batch_size, d_model)
            target_index index, size of (batch_size, )
        """
        batch_size = target_index.size(0)

        h = torch.reshape(h, (batch_size, 1, self.d_model))

        # positive
        pos_out = torch.sigmoid(
            self.embedding.forward(h, torch.reshape(target_index, (batch_size, 1)))
        )
        pos_out = torch.reshape(pos_out, (batch_size, 1))
        pos_label = torch.ones(batch_size, 1)

        # negative
        # (batch_size, negative_sample_size)
        negative_sample = torch.tensor(
            self.sampler.get_negative_sample(batch_size, self.negative_sample_size),
            dtype=torch.long,
        )

        neg_out = torch.sigmoid(self.embedding.forward(h, negative_sample))
        neg_out = torch.reshape(neg_out, (batch_size, self.negative_sample_size))
        neg_label = torch.zeros(batch_size, self.negative_sample_size)

        return pos_out, pos_label, neg_out, neg_label


class MyEmbedding(nn.Module):
    """
    nn.Embeddingのラッパークラス
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        mean: float = 0,
        std: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            max_norm=max_norm,
        )
        nn.init.normal_(self.embedding.weight, mean=mean, std=std)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding.forward(x)

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight


class WeightSharedNegativeSampling(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_item_meta_types: int,
        sequences: List[List[int]],
        item_meta_indicies: Tensor,
        item_meta_weights: Tensor,
        embedding_item: MyEmbedding,
        embedding_item_meta: MyEmbedding,
        power: float = 0.75,
        negative_sample_size: int = 5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_item_meta_types = num_item_meta_types
        self.negative_sample_size = negative_sample_size
        self.item_meta_indicies = item_meta_indicies
        self.item_meta_weights = item_meta_weights
        self.embedding_item = embedding_item
        self.embedding_item_meta = embedding_item_meta
        self.sampler = UnigramSampler(sequences, power)

    def forward(
        self,
        h: Tensor,
        target_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            h (Tensor): (batch_size, d_model)
            target_index (Tensor): (batch_size, )

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                pos_out: (batch_size, ),
                pos_label: (batch_size, ),
                neg_out: (batch_size, negative_sample_size),
                neg_label: (batch_size, negative_sample_size)
        """
        batch_size = target_index.size(0)

        h = torch.reshape(h, (batch_size, 1, self.d_model))

        item_meta_index = self.item_meta_indicies[target_index]
        item_meta_weight = self.item_meta_weights[target_index]

        # positive
        h_items = self.embedding_item.forward(target_index)
        # add meta embedding
        h_item_meta = self.embedding_item_meta.forward(item_meta_index)
        h_item_meta_weighted = calc_weighted_meta(h_item_meta, item_meta_weight)
        h_items += h_item_meta_weighted
        # take mean
        h_items /= self.num_item_meta_types + 1
        h_items = torch.reshape(h_items, (-1, 1, self.d_model))
        pos_out = torch.sigmoid(torch.matmul(h, h_items.mT))
        pos_out = torch.reshape(pos_out, (batch_size, 1))
        pos_label = torch.ones(batch_size, 1)

        # negative
        # (batch_size, negative_sample_size)
        negative_sample = torch.tensor(
            self.sampler.get_negative_sample(batch_size, self.negative_sample_size),
            dtype=torch.long,
        )
        item_meta_index = self.item_meta_indicies[negative_sample]
        item_meta_weight = self.item_meta_weights[negative_sample]
        h_items = self.embedding_item.forward(negative_sample)
        # add meta embedding
        h_item_meta = self.embedding_item_meta.forward(item_meta_index)
        h_item_meta_weighted = calc_weighted_meta(h_item_meta, item_meta_weight)
        h_items += h_item_meta_weighted
        # take mean
        h_items /= self.num_item_meta_types + 1
        h_items = torch.reshape(h_items, (-1, self.negative_sample_size, self.d_model))
        neg_out = torch.sigmoid(torch.matmul(h, h_items.mT))
        neg_out = torch.reshape(neg_out, (batch_size, self.negative_sample_size))
        neg_label = torch.zeros(batch_size, self.negative_sample_size)

        return pos_out, pos_label, neg_out, neg_label


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, max_sequence_length: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_sequence_length, d_model)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # d_model needs to be a even number
        assert d_model % 2 == 0
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe_const", self.pe)

    def forward(self, hs: Tensor) -> Tensor:
        """Add positional encoding

        Args:
            hs (Tensor):
                size: (batch_size, sequence_length, d_model)

        Returns:
            Tensor: positional encoded hs
        """
        hs = hs + self.pe[: hs.size(0), :]
        return self.dropout.forward(hs)
