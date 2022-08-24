from math import sqrt
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from layer import NegativeSampling


def attention_weight(Q: Tensor, K: Tensor) -> Tensor:
    dim = len(Q.shape) - 1  # to handle batched and unbatched data
    return F.softmax(torch.matmul(Q, K.mT) / sqrt(K.size(dim)), dim=dim)


def attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    a = attention_weight(Q, K)
    return torch.matmul(a, V)


class AttentiveModel(nn.Module):
    def __init__(
        self,
        num_seq: int,
        num_item: int,
        num_meta: int,
        d_model: int,
        sequences: List[List[int]],
        negative_sample_size: int = 30,
        model_path: Optional[str] = None,
        pretrained_embedding: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.embedding_seq = nn.Embedding(num_seq, d_model)
        self.embedding_item = nn.Embedding(num_item, d_model)
        self.embedding_meta = nn.Embedding(num_meta, d_model)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)

        self.output = NegativeSampling(
            d_model=d_model,
            num_item=num_item,
            sequences=sequences,
            negative_sample_size=negative_sample_size,
        )

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))  # type: ignore
        elif pretrained_embedding is not None:
            item_embedding, seq_embedding = pretrained_embedding
            self.seq_embedding.copy_(seq_embedding)
            self.item_embedding.copy_(item_embedding)

    def forward(
        self,
        seq_index: Tensor,
        item_indicies: Tensor,
        meta_indicies: Tensor,
        target_index: Tensor,
    ) -> Tensor:
        r"""
        seq_index:
            type: `int` or `long`
            shape: (batch_size, )
        item_indicies:
            type: `int` or `long`
            shape: (batch_size, window_size)
        meta_indicies:
            type: `int` or `long`
            shape: (batch_size, window_size, num_meta_types)
        """
        num_meta_types = meta_indicies.size(2)

        h_seq = self.embedding_seq.forward(seq_index)
        h_items = self.embedding_item.forward(item_indicies)
        # add meta embedding
        h_items += self.embedding_meta.forward(meta_indicies).sum(dim=2)
        # take mean
        h_items /= num_meta_types

        Q = torch.reshape(self.W_q(h_seq), (-1, 1, self.d_model))
        K = self.W_k(h_items)
        V = h_items

        c = torch.reshape(attention(Q, K, V), (-1, self.d_model))
        v = (c + h_seq) / 2

        loss = self.output.forward(v, target_index)
        return loss

    @property
    def seq_embedding(self) -> Tensor:
        return self.embedding_seq.weight.data

    @property
    def item_embedding(self) -> Tensor:
        return self.embedding_item.weight.data


class OriginalDoc2Vec(nn.Module):
    def __init__(
        self, num_seq: int, num_item: int, d_model: int, sequences: List[List[int]]
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.W_seq = nn.Embedding(num_seq, d_model)
        self.W_item = nn.Embedding(num_item, d_model)

        self.projection = nn.Linear(d_model, num_item)

    def forward(
        self, seq_index: Tensor, item_indicies: Tensor, target_index: Tensor
    ) -> Tensor:
        r"""
        seq_index:
            type: `int` or `long`
            shape: (batch_size, )
        item_indicies:
            type: `int` or `long`
            shape: (batch_size, window_size)
        """
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
