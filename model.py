import abc
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from layer import (
    NegativeSampling,
    PositionalEncoding,
    attention,
    attention_weight,
    calc_weighted_meta,
    cosine_similarity,
)


class Model(metaclass=abc.ABCMeta):
    def forward(
        self,
        seq_index: Tensor,
        item_indicies: Tensor,
        seq_meta_indicies: Tensor,
        item_meta_indicies: Tensor,
        item_meta_weights: Tensor,
        target_index: Tensor,
    ) -> Tensor:
        """
        モデルに入力を与えた時の損失

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
                size: (batch_size, window_size, max_item_meta_size, )
            item_meta_weights (Tensor):
                要素の補助情報の重み
                一つの補助情報に対して、複数の指定がある場合に、重みをかけて平均を取る
                例:
                ["color:blue", "color:dark", "genre:shirt"]
                [0.5, 0.5, 1]
                size: (batch_size, window_size, max_item_meta_size, )
            target_index (Tensor):
                size: (batch_size, )

        Returns:
            loss: Tensor
        """
        pos_out, pos_label, neg_out, neg_label = self.calc_out(
            seq_index=seq_index,
            item_indicies=item_indicies,
            seq_meta_indicies=seq_meta_indicies,
            item_meta_indicies=item_meta_indicies,
            item_meta_weights=item_meta_weights,
            target_index=target_index,
        )
        loss_pos = F.binary_cross_entropy(pos_out, pos_label)
        loss_neg = F.binary_cross_entropy(neg_out, neg_label)

        negative_sample_size = neg_label.size(1)
        loss = (loss_pos + loss_neg / negative_sample_size) / 2

        return loss

    @abc.abstractmethod
    def calc_out(
        self,
        seq_index: Tensor,
        item_indicies: Tensor,
        seq_meta_indicies: Tensor,
        item_meta_indicies: Tensor,
        item_meta_weights: Tensor,
        target_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        モデルに入力を与えた時の、損失を求める直前の出力を返す
        正例と負例に対する0~1の（シグモイドを通した）出力をする
        `forward`の中で使われることを想定している

        Args:
            Same as `Model.forward()`

        Returns:
            (pos_out, pos_label, neg_out, neg_label)
                : Tuple[Tensor, Tensor, Tensor, Tensor]
                pos_out: (batch_size, ),
                pos_label: (batch_size, ),
                neg_out: (batch_size, negative_sample_size),
                neg_label: (batch_size, negative_sample_size),
        """
        raise NotImplementedError()

    @torch.no_grad()  # type: ignore
    def similarity_between_seq_and_item_meta(
        self, seq_index: int, meta_indicies: List[int], method: str = "attention"
    ) -> Tensor:
        raise NotImplementedError(
            "similarity_between_seq_and_item_meta is not supported for "
            + f"{self.__class__.__name__}"
        )

    @torch.no_grad()  # type: ignore
    def similarity_between_seq_and_item(
        self, seq_index: int, item_indicies: List[int], method: str = "attention"
    ) -> Tensor:
        raise NotImplementedError(
            "similarity_between_seq_and_item is not supported for "
            + f"{self.__class__.__name__}"
        )

    @torch.no_grad()  # type: ignore
    def similarity_between_seq_meta_and_item_meta(
        self,
        seq_meta_index: int,
        item_meta_indicies: List[int],
        method: str = "attention",
    ) -> Tensor:
        raise NotImplementedError(
            "similarity_between_seq_meta_and_item_meta is not supported for "
            + f"{self.__class__.__name__}"
        )

    @abc.abstractproperty
    @property
    def seq_embedding(self) -> Tensor:
        raise NotImplementedError()

    @abc.abstractproperty
    @property
    def item_embedding(self) -> Tensor:
        raise NotImplementedError()

    @property
    def seq_meta_embedding(self) -> Tensor:
        raise NotImplementedError(
            "seq_meta_embedding is not supported for " + f"{self.__class__.__name__}"
        )

    @property
    def item_meta_embedding(self) -> Tensor:
        raise NotImplementedError(
            "item_meta_embedding is not supported for " + f"{self.__class__.__name__}"
        )


class PyTorchModel(Model, nn.Module):
    pass


class AttentiveModel(PyTorchModel):
    """AttentiveModel（提案モデル）のクラス"""

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
        add_seq_embedding: bool = False,
        add_positional_encoding: bool = False,
    ) -> None:
        """
        AttentiveModel（提案モデル）のクラスを生成する

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

        self.output = NegativeSampling(
            d_model=d_model,
            num_item=num_item,
            sequences=sequences,
            negative_sample_size=negative_sample_size,
        )

    def calc_out(
        self,
        seq_index: Tensor,
        item_indicies: Tensor,
        seq_meta_indicies: Tensor,
        item_meta_indicies: Tensor,
        item_meta_weights: Tensor,
        target_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
        h_item_meta = self.embedding_item_meta.forward(item_meta_indicies)
        h_item_meta_weighted = calc_weighted_meta(h_item_meta, item_meta_weights)
        h_items += h_item_meta_weighted
        # take mean
        h_items /= num_item_meta_types + 1

        if self.add_positional_encoding:
            h_items = self.positional_encoding.forward(h_items)

        Q = torch.reshape(h_seq, (-1, 1, self.d_model))
        K = h_items
        V = h_items
        c = torch.reshape(attention(Q, K, V), (-1, self.d_model))

        if self.add_seq_embedding:
            v = (c * window_size + h_seq) / (window_size + 1)
        else:
            v = c

        return self.output.forward(v, target_index)

    @torch.no_grad()  # type: ignore
    def similarity_between_seq_and_item_meta(
        self, seq_index: int, item_meta_indicies: List[int], method: str = "attention"
    ) -> Tensor:
        seq_index = torch.LongTensor([seq_index])
        item_meta_indicies = torch.LongTensor(item_meta_indicies)
        h_seq = self.embedding_seq.forward(seq_index)
        h_item_meta = self.embedding_item_meta.forward(item_meta_indicies)
        match method:
            case "attention":
                attention_weight(h_seq, h_item_meta)
            case "cos":
                weight = cosine_similarity(h_seq, h_item_meta)
            case "inner-product":
                weight = torch.matmul(h_seq, h_item_meta.mT)
            case _:
                assert False, f"Invalid method {method}"
        return weight.squeeze()

    @torch.no_grad()  # type: ignore
    def similarity_between_seq_and_item(
        self, seq_index: int, item_indicies: List[int], method: str = "attention"
    ) -> Tensor:
        seq_index = torch.LongTensor([seq_index])
        item_indicies = torch.LongTensor(item_indicies)
        h_seq = self.embedding_seq.forward(seq_index)
        h_item = self.embedding_item.forward(item_indicies)
        match method:
            case "attention":
                weight = attention_weight(h_seq, h_item)
            case "cos":
                weight = cosine_similarity(h_seq, h_item)
            case "inner-product":
                weight = torch.matmul(h_item, h_seq.mT)
            case _:
                assert False, f"Invalid method {method}"
        return weight.squeeze()

    @torch.no_grad()  # type: ignore
    def similarity_between_seq_meta_and_item_meta(
        self,
        seq_meta_index: int,
        item_meta_indicies: List[int],
        method: str = "attention",
    ) -> Tensor:
        seq_meta_index = torch.LongTensor(seq_meta_index)
        item_meta_indicies = torch.LongTensor(item_meta_indicies)
        h_seq_meta = self.embedding_seq_meta.forward(seq_meta_index)
        h_item_meta = self.embedding_item_meta.forward(item_meta_indicies)

        match method:
            case "attention":
                weight = attention_weight(h_seq_meta, h_item_meta)
            case "cos":
                weight = cosine_similarity(h_seq_meta, h_item_meta)
            case "inner-product":
                weight = torch.matmul(h_item_meta, h_seq_meta.mT)
            case _:
                assert False, f"Invalid method {method}"
        return weight.squeeze()

    @property
    def seq_embedding(self) -> Tensor:
        return self.embedding_seq.weight.data

    @property
    def item_embedding(self) -> Tensor:
        return self.embedding_item.weight.data

    @property
    def seq_meta_embedding(self) -> Tensor:
        return self.embedding_seq_meta.weight.data

    @property
    def item_meta_embedding(self) -> Tensor:
        return self.embedding_item_meta.weight.data


class Doc2Vec(PyTorchModel):
    """Original Doc2Vec"""

    def __init__(
        self,
        num_seq: int,
        num_item: int,
        d_model: int,
        sequences: List[List[int]],
        negative_sample_size: int = 30,
    ) -> None:
        """
        Original Doc2Vecを生成する

        Args:
            num_seq (int):
                系列の総数
            num_item (int):
                要素の総数
            d_model (int):
                埋め込み表現の次元数
            sequences (List[List[int]]):
                変換後の系列データ
            negative_sample_size (int, optional):
                ネガティブサンプリングのサンプリング数. Defaults to 30.
        """
        super().__init__()

        self.embedding_seq = nn.Embedding(num_seq, d_model)
        self.embedding_item = nn.Embedding(num_item, d_model)

        self.output = NegativeSampling(
            d_model=d_model,
            num_item=num_item,
            sequences=sequences,
            negative_sample_size=negative_sample_size,
        )

    def calc_out(
        self,
        seq_index: Tensor,
        item_indicies: Tensor,
        seq_meta_indicies: Tensor,
        item_meta_indicies: Tensor,
        item_meta_weights: Tensor,
        target_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        window_size = item_indicies.size(1)

        h_seq = self.embedding_seq.forward(seq_index)
        h_items = self.embedding_item.forward(item_indicies)

        v = (h_seq + h_items.sum(dim=1)) / (window_size + 1)

        return self.output.forward(v, target_index)

    @property
    def seq_embedding(self) -> Tensor:
        return self.embedding_seq.weight.data

    @property
    def item_embedding(self) -> Tensor:
        return self.embedding_item.weight.data
