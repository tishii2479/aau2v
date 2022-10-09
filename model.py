import abc
from math import sqrt
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from layer import NegativeSampling, PositionalEncoding


class Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calc_out(
        self,
        seq_index: Tensor,
        item_indicies: Tensor,
        seq_meta_indicies: Tensor,
        item_meta_indicies: Tensor,
        target_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError()

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
        num_seq_meta: int,
        num_item_meta: int,
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
        self.embedding_seq_meta = nn.Embedding(num_seq_meta, d_model)
        self.embedding_item_meta = nn.Embedding(num_item_meta, d_model)
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
        item_meta_indicies: Tensor,
        seq_meta_indicies: Tensor,
        target_index: Tensor,
    ) -> Tensor:
        r"""
        Args:
            seq_index (Tensor):
                学習対象である系列のindex
                size: (batch_size, )
            item_indicies (Tensor):
                予測に用いる直前の要素のindicies
                size: (batch_size, window_size, )
            seq_meta_indicies (Tensor):
                系列の補助情報のindicies
                size: (batch_size, seq_meta_kinds, )
            item_meta_indicies (Tensor):
                要素の補助情報のindicies
                size: (batch_size, window_size, item_meta_kinds, )
            target_index (Tensor):
                size: (batch_size, )

        Returns:
            loss
        """
        pos_out, pos_label, neg_out, neg_label = self.calc_out(
            seq_index=seq_index,
            item_indicies=item_indicies,
            seq_meta_indicies=seq_meta_indicies,
            item_meta_indicies=item_meta_indicies,
            target_index=target_index,
        )
        pos_loss = F.binary_cross_entropy(pos_out, pos_label)
        neg_loss = F.binary_cross_entropy(neg_out, neg_label)

        negative_sample_size = neg_label.size(1)
        loss = (pos_loss + neg_loss / negative_sample_size) / 2

        return loss

    def calc_out(
        self,
        seq_index: Tensor,
        item_indicies: Tensor,
        seq_meta_indicies: Tensor,
        item_meta_indicies: Tensor,
        target_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        正例と負例に対する0~1の（シグモイドを通した）出力をする

        Args:
            seq_index (Tensor):
                学習対象である系列のindex
                size: (batch_size, )
            item_indicies (Tensor):
                予測に用いる直前の要素のindicies
                size: (batch_size, window_size, )
            seq_meta_indicies (Tensor):
                系列の補助情報のindicies
                size: (batch_size, seq_meta_kinds, )
            item_meta_indicies (Tensor):
                要素の補助情報のindicies
                size: (batch_size, window_size, item_meta_kinds, )
            target_index (Tensor):
                size: (batch_size, )

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                pos_out: (batch_size, ),
                pos_label: (batch_size, ),
                neg_out: (batch_size, negative_sample_size),
                neg_label: (batch_size, negative_sample_size),
        """
        num_seq_meta_types = seq_meta_indicies.size(1)
        num_item_meta_types = item_meta_indicies.size(2)
        window_size = item_indicies.size(1)

        h_seq = self.embedding_seq.forward(seq_index)
        # add meta embedding
        h_seq += self.embedding_seq_meta.forward(seq_meta_indicies).sum(dim=1)
        # take mean
        h_seq /= num_seq_meta_types + 1

        h_items = self.embedding_item.forward(item_indicies)
        # add meta embedding
        h_items += self.embedding_item_meta.forward(item_meta_indicies).sum(dim=2)
        # take mean
        h_items /= num_item_meta_types + 1

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

        return self.output.forward(v, target_index)

    @torch.no_grad()  # type: ignore
    def attention_weight_to_meta(
        self,
        seq_index: int,
        meta_indicies: List[int],
    ) -> Tensor:
        seq_index = torch.LongTensor([seq_index])
        meta_indicies = torch.LongTensor(meta_indicies)
        h_seq = self.embedding_seq.forward(seq_index)
        h_meta = self.embedding_item_meta.forward(meta_indicies)
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
