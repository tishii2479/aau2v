import abc
from math import sqrt
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from layer import NegativeSampling, PositionalEncoding


class Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    @torch.no_grad()  # type: ignore
    def attention_weight_to_meta(
        self, seq_index: int, meta_indicies: List[int]
    ) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    @torch.no_grad()  # type: ignore
    def attention_weight_to_item(
        self, seq_index: int, item_indicies: List[int]
    ) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    @torch.no_grad()  # type: ignore
    def attention_weight_to_positional_encoding(
        self, seq_index: int, item_indicies: List[int]
    ) -> Tensor:
        raise NotImplementedError()

    @abc.abstractproperty
    @property
    def seq_embedding(self) -> Tensor:
        raise NotImplementedError()

    @abc.abstractproperty
    @property
    def item_embedding(self) -> Tensor:
        raise NotImplementedError()


def attention_weight(Q: Tensor, K: Tensor) -> Tensor:
    dim = len(Q.shape) - 1  # to handle batched and unbatched data
    return F.softmax(torch.matmul(Q, K.mT) / sqrt(K.size(dim)), dim=dim)


def attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    a = attention_weight(Q, K)
    return torch.matmul(a, V)


class PyTorchModel(nn.Module, Model):
    pass


class AttentiveModel(PyTorchModel):
    def __init__(
        self,
        num_seq: int,
        num_item: int,
        num_meta: int,
        d_model: int,
        sequences: List[List[int]],
        negative_sample_size: int = 30,
        max_sequence_length: int = 1000,
        dropout: float = 0.1,
        add_seq_embedding: bool = True,
        add_positional_encoding: bool = False,
    ) -> None:
        """
        AttentiveModel（提案モデル）のクラス

        Args:
            num_seq (int):
                系列の総数
            num_item (int):
                要素の総数
            num_meta (int):
                要素の補助情報の総数
            d_model (int):
                埋め込み表現の次元数
            sequences (List[List[int]]):
                変換後の系列データ
            negative_sample_size (int, optional):
                ネガティブサンプリングのサンプリング数. Defaults to 30.
            max_sequence_length (int, optional):
                系列の最大長. Defaults to 1000.
            dropout (float, optional):
                位置エンコーディング時にドロップアウトする比率. Defaults to 0.1.
            add_seq_embedding (bool, optional):
                系列の埋め込み表現を予測ベクトルに足すかどうか. Defaults to True.
            add_positional_encoding (bool, optional):
                位置エンコーディングを行うかどうか. Defaults to False.
        """
        super().__init__()
        self.d_model = d_model

        self.embedding_seq = nn.Embedding(num_seq, d_model)
        self.embedding_item = nn.Embedding(num_item, d_model)
        self.embedding_meta = nn.Embedding(num_meta, d_model)
        self.add_seq_embedding = add_seq_embedding
        self.add_positional_encoding = add_positional_encoding

        if self.add_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model, max_sequence_length, dropout
            )

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)

        self.output = NegativeSampling(
            d_model=d_model,
            num_item=num_item,
            sequences=sequences,
            negative_sample_size=negative_sample_size,
        )

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
        window_size = item_indicies.size(1)

        h_seq = self.embedding_seq.forward(seq_index)
        h_items = self.embedding_item.forward(item_indicies)
        # add meta embedding
        h_items += self.embedding_meta.forward(meta_indicies).sum(dim=2)
        # take mean
        h_items /= num_meta_types + 1

        if self.add_positional_encoding:
            h_items = self.positional_encoding.forward(h_items)

        Q = torch.reshape(self.W_q(h_seq), (-1, 1, self.d_model))
        K = self.W_k(h_items)
        V = h_items
        c = torch.reshape(attention(Q, K, V), (-1, self.d_model))

        if self.add_seq_embedding:
            v = (c * window_size + h_seq) / (window_size + 1)
        else:
            v = c

        loss = self.output.forward(v, target_index)
        return loss

    @torch.no_grad()  # type: ignore
    def attention_weight_to_meta(
        self,
        seq_index: int,
        meta_indicies: List[int],
    ) -> Tensor:
        seq_index = torch.LongTensor([seq_index])
        meta_indicies = torch.LongTensor(meta_indicies)
        h_seq = self.embedding_seq.forward(seq_index)
        h_meta = self.embedding_meta.forward(meta_indicies)
        weight = attention_weight(h_seq, h_meta)
        return weight

    @torch.no_grad()  # type: ignore
    def attention_weight_to_item(
        self,
        seq_index: int,
        item_indicies: List[int],
    ) -> Tensor:
        seq_index = torch.LongTensor([seq_index])
        item_indicies = torch.LongTensor(item_indicies)
        h_seq = self.embedding_seq.forward(seq_index)
        h_item = self.embedding_item.forward(item_indicies)
        weight = attention_weight(h_seq, h_item)
        return weight

    @torch.no_grad()  # type: ignore
    def attention_weight_to_positional_encoding(
        self, seq_index: int, item_indicies: List[int]
    ) -> Tensor:
        raise NotImplementedError()

    @property
    def seq_embedding(self) -> Tensor:
        return self.embedding_seq.weight.data

    @property
    def item_embedding(self) -> Tensor:
        return self.embedding_item.weight.data
