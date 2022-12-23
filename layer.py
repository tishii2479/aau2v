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


def calc_weighted_meta(e_meta: Tensor, meta_weights: Tensor) -> Tensor:
    return torch.matmul(
        e_meta.mT,
        meta_weights.view((*e_meta.shape[:-1], 1)),
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
        self,
        d_model: int,
        num_item: int,
        max_embedding_norm: Optional[float] = None,
        init_embedding_std: float = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = NormalizedEmbeddingLayer(
            num_item, d_model, max_norm=max_embedding_norm, std=init_embedding_std
        )

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
        init_embedding_std: float = 1,
        power: float = 0.75,
        negative_sample_size: int = 5,
        max_embedding_norm: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.negative_sample_size = negative_sample_size
        self.sampler = UnigramSampler(sequences, power)
        self.embedding = EmbeddingDot(
            d_model,
            num_item,
            max_embedding_norm=max_embedding_norm,
            init_embedding_std=init_embedding_std,
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


class NormalizedEmbeddingLayer(nn.Embedding):
    """
    nn.Embeddingのラッパークラス
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        normalize_weight: bool = True,
        max_norm: Optional[float] = None,
        mean: float = 0,
        std: float = 1,
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            max_norm=max_norm,
        )
        nn.init.normal_(self.weight, mean=mean, std=std)
        self.normalize_weight = normalize_weight
        self.forward_count = 0

    def forward(self, x: Tensor) -> Tensor:
        # ISSUE: 補助情報ごとに正規化してみても良さそう
        if self.normalize_weight and torch.is_grad_enabled():
            self.forward_count += 1
            # 毎回重みを正規化すると以下の問題があるため、1024回forwardが呼ばれるたびに正規化する
            # - 学習時間が長くなる
            # - 学習率が高いと、学習量が各要素間で偏っている状態で正規化してしまうためうまくいかない
            # ISSUE: self.forward_countが本当に必要か確認する
            if self.forward_count % 1024 == 0:
                with torch.no_grad():
                    # 埋め込み表現の各次元の大きさの最大値を1にする
                    # TODO: 最大値を注入できるようにする
                    self.weight.data /= self.weight.abs().max()
        return super().forward(x)


class MetaEmbeddingLayer(nn.Module):
    def __init__(
        self,
        num_element: int,
        num_meta: int,
        num_meta_types: int,
        d_model: int,
        meta_indicies: Tensor,
        meta_weights: Tensor,
        normalize_embedding_dim: bool = True,
        max_embedding_norm: Optional[float] = None,
        init_embedding_std: float = 1,
    ):
        super().__init__()
        self.embedding_element = NormalizedEmbeddingLayer(
            num_element,
            d_model,
            max_norm=max_embedding_norm,
            std=init_embedding_std,
            normalize_weight=normalize_embedding_dim,
        )
        self.embedding_meta = NormalizedEmbeddingLayer(
            num_meta,
            d_model,
            max_norm=max_embedding_norm,
            std=init_embedding_std,
            normalize_weight=normalize_embedding_dim,
        )
        self.num_meta_types = num_meta_types
        self.meta_indicies = meta_indicies
        self.meta_weights = meta_weights

    def forward(self, element_indicies: Tensor) -> Tensor:
        e_element = self.embedding_element.forward(element_indicies)
        # add meta embedding
        meta_index = self.meta_indicies[element_indicies]
        meta_weight = self.meta_weights[element_indicies]
        e_meta = self.embedding_meta.forward(meta_index)
        e_meta_weighted = calc_weighted_meta(e_meta, meta_weight)
        e_element += e_meta_weighted
        # take mean
        e_element /= self.num_meta_types + 1
        return e_element


class WeightSharedNegativeSampling(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_item_meta_types: int,
        sequences: List[List[int]],
        item_meta_indicies: Tensor,
        item_meta_weights: Tensor,
        embedding_item: nn.Module,
        power: float = 0.75,
        negative_sample_size: int = 5,
    ) -> None:
        # TODO: write doc
        super().__init__()
        self.d_model = d_model
        self.num_item_meta_types = num_item_meta_types
        self.negative_sample_size = negative_sample_size
        self.item_meta_indicies = item_meta_indicies
        self.item_meta_weights = item_meta_weights
        self.embedding_item = embedding_item
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

        # positive
        e_pos_items = self.embedding_item.forward(target_index)
        e_pos_items = torch.reshape(e_pos_items, (-1, 1, self.d_model))
        pos_out = torch.sigmoid(torch.matmul(h, e_pos_items.mT))
        pos_out = torch.reshape(pos_out, (batch_size, 1))
        pos_label = torch.ones(batch_size, 1)

        # negative
        # (batch_size, negative_sample_size)
        negative_sample = torch.tensor(
            self.sampler.get_negative_sample(batch_size, self.negative_sample_size),
            dtype=torch.long,
        )
        e_neg_items = self.embedding_item.forward(negative_sample)
        e_neg_items = torch.reshape(
            e_neg_items, (-1, self.negative_sample_size, self.d_model)
        )
        neg_out = torch.sigmoid(torch.matmul(h, e_neg_items.mT))
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
