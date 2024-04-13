import collections
from math import sqrt
from typing import Optional, Tuple

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
    def __init__(
        self, vocab_size: int, counter: collections.Counter, power: float
    ) -> None:
        self.vocab_size = vocab_size
        self.word_p = np.zeros(vocab_size)
        for i in range(self.vocab_size):
            self.word_p[i] = max(1, counter[i])

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
        self.embedding = EmbeddingLayer(
            num_item, d_model, max_norm=max_embedding_norm, std=init_embedding_std
        )

    def forward(self, h: Tensor, indices: Tensor) -> Tensor:
        """Forward Embedding Dot

        Args:
            h (Tensor): input, size of (batch_size, 1, d_model)
            indices (Tensor): indices selected to get embedding,
                size of (batch_size, 1 or sample_size)

        Returns:
            Tensor: output
        """
        w = self.embedding.forward(indices)
        w = torch.reshape(w, (-1, indices.size(1), self.d_model))
        out = torch.matmul(h, w.mT)
        return out


class NegativeSampling(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_item: int,
        item_counter: collections.Counter,
        device: str = "cpu",
        init_embedding_std: float = 1,
        power: float = 0.75,
        negative_sample_size: int = 5,
        max_embedding_norm: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.negative_sample_size = negative_sample_size
        self.sampler = UnigramSampler(
            vocab_size=num_item, counter=item_counter, power=power
        )
        self.embedding = EmbeddingDot(
            d_model,
            num_item,
            max_embedding_norm=max_embedding_norm,
            init_embedding_std=init_embedding_std,
        )
        self.device = device

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
        pos_label = torch.ones(batch_size, 1).to(device=self.device)

        # negative
        # (batch_size, negative_sample_size)
        negative_sample = torch.tensor(
            self.sampler.get_negative_sample(batch_size, self.negative_sample_size),
            dtype=torch.long,
        ).to(device=self.device)

        neg_out = torch.sigmoid(self.embedding.forward(h, negative_sample))
        neg_out = torch.reshape(neg_out, (batch_size, self.negative_sample_size))
        neg_label = torch.zeros(batch_size, self.negative_sample_size).to(
            device=self.device
        )

        return pos_out, pos_label, neg_out, neg_label


class EmbeddingLayer(nn.Embedding):
    """
    nn.Embeddingのラッパークラス
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: Optional[float] = None,
        mean: float = 0,
        std: float = 0.2,
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            max_norm=max_norm,
            scale_grad_by_freq=True,
        )
        nn.init.normal_(self.weight, mean=mean, std=std)


class MetaEmbeddingLayer(nn.Module):
    def __init__(
        self,
        num_element: int,
        num_meta: int,
        num_meta_types: int,
        d_model: int,
        meta_indices: Tensor,
        meta_weights: Tensor,
        max_embedding_norm: Optional[float] = None,
        init_embedding_std: float = 0.2,
    ):
        super().__init__()
        self.embedding_element = EmbeddingLayer(
            num_element,
            d_model,
            max_norm=max_embedding_norm,
            std=init_embedding_std,
        )
        self.embedding_meta = EmbeddingLayer(
            num_meta,
            d_model,
            max_norm=max_embedding_norm,
            std=init_embedding_std,
        )
        self.num_meta_types = num_meta_types
        self.meta_indices = meta_indices
        self.meta_weights = meta_weights

    def forward(self, element_indices: Tensor) -> Tensor:
        e_element = self.embedding_element.forward(element_indices)
        # add meta embedding
        meta_index = self.meta_indices[element_indices]
        meta_weight = self.meta_weights[element_indices]
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
        num_item: int,
        num_item_meta_types: int,
        item_counter: collections.Counter,
        item_meta_indices: Tensor,
        item_meta_weights: Tensor,
        embedding_item: nn.Module,
        device: str = "cpu",
        power: float = 0.75,
        negative_sample_size: int = 5,
    ) -> None:
        # TODO: write doc
        super().__init__()
        self.d_model = d_model
        self.num_item_meta_types = num_item_meta_types
        self.negative_sample_size = negative_sample_size
        self.item_meta_indices = item_meta_indices
        self.item_meta_weights = item_meta_weights
        self.embedding_item = embedding_item
        self.device = device
        self.sampler = UnigramSampler(
            vocab_size=num_item, counter=item_counter, power=power
        )

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
        pos_label = torch.ones(batch_size, 1).to(self.device)

        # negative
        # (batch_size, negative_sample_size)
        negative_sample = torch.tensor(
            self.sampler.get_negative_sample(batch_size, self.negative_sample_size),
            dtype=torch.long,
        ).to(self.device)
        e_neg_items = self.embedding_item.forward(negative_sample)
        e_neg_items = torch.reshape(
            e_neg_items, (-1, self.negative_sample_size, self.d_model)
        )
        neg_out = torch.sigmoid(torch.matmul(h, e_neg_items.mT))
        neg_out = torch.reshape(neg_out, (batch_size, self.negative_sample_size))
        neg_label = torch.zeros(batch_size, self.negative_sample_size).to(self.device)

        return pos_out, pos_label, neg_out, neg_label
