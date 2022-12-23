import abc
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from layer import (
    MetaEmbeddingLayer,
    NegativeSampling,
    NormalizedEmbeddingLayer,
    WeightSharedNegativeSampling,
    attention,
)


class Model(metaclass=abc.ABCMeta):
    def forward(
        self,
        seq_index: Tensor,
        item_indices: Tensor,
        target_index: Tensor,
    ) -> Tensor:
        """
        モデルに入力を与えた時の損失

        Args:
            seq_index (Tensor):
                学習対象である系列のindex
                size: (batch_size, )
            item_indices (Tensor):
                予測に用いる直前の要素のindices
                size: (batch_size, window_size, )
            target_index (Tensor):
                size: (batch_size, )

        Returns:
            loss: Tensor
        """
        pos_out, pos_label, neg_out, neg_label = self.calc_out(
            seq_index=seq_index,
            item_indices=item_indices,
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
        item_indices: Tensor,
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

    @abc.abstractmethod
    def calc_prediction_vector(
        self,
        seq_index: Tensor,
        item_indices: Tensor,
    ) -> Tensor:
        """
        モデルに入力を与えた時の、出力層に入力する前の予測ベクトルを返す
        `calc_out`の中で使われることを想定している

        Args:
            Same as `Model.forward()`

        Returns:
            p: 予測ベクトル (batch_size, d_model)
        """
        raise NotImplementedError()

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

    @property
    def output_item_embedding(self) -> Tensor:
        raise NotImplementedError()


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
        num_seq_meta_types: int,
        num_item_meta_types: int,
        sequences: List[List[int]],
        seq_meta_indices: Tensor,
        seq_meta_weights: Tensor,
        item_meta_indices: Tensor,
        item_meta_weights: Tensor,
        d_model: int = 128,
        init_embedding_std: float = 1,
        normalize_embedding_dim: bool = True,
        max_embedding_norm: Optional[float] = None,
        negative_sample_size: int = 5,
    ) -> None:
        """
        TODO: 書き直す
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
        """
        super().__init__()
        self.d_model = d_model

        self.embedding_seq = MetaEmbeddingLayer(
            num_element=num_seq,
            num_meta=num_seq_meta,
            num_meta_types=num_seq_meta_types,
            d_model=d_model,
            meta_indices=seq_meta_indices,
            meta_weights=seq_meta_weights,
            normalize_embedding_dim=normalize_embedding_dim,
            max_embedding_norm=max_embedding_norm,
            init_embedding_std=init_embedding_std,
        )
        self.embedding_item = MetaEmbeddingLayer(
            num_element=num_item,
            num_meta=num_item_meta,
            num_meta_types=num_item_meta_types,
            d_model=d_model,
            meta_indices=item_meta_indices,
            meta_weights=item_meta_weights,
            normalize_embedding_dim=normalize_embedding_dim,
            max_embedding_norm=max_embedding_norm,
            init_embedding_std=init_embedding_std,
        )

        self.output = WeightSharedNegativeSampling(
            d_model=d_model,
            num_item_meta_types=num_item_meta_types,
            sequences=sequences,
            negative_sample_size=negative_sample_size,
            item_meta_indices=item_meta_indices,
            item_meta_weights=item_meta_weights,
            embedding_item=self.embedding_item,
        )

    def calc_out(
        self,
        seq_index: Tensor,
        item_indices: Tensor,
        target_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        c = self.calc_prediction_vector(
            seq_index=seq_index,
            item_indices=item_indices,
        )
        return self.output.forward(c, target_index)

    def calc_prediction_vector(
        self,
        seq_index: Tensor,
        item_indices: Tensor,
    ) -> Tensor:
        u = self.embedding_seq.forward(seq_index)
        V = self.embedding_item.forward(item_indices)

        Q = torch.reshape(u, (-1, 1, self.d_model))
        K = V
        V = V
        p = torch.reshape(attention(Q, K, V), (-1, self.d_model))

        return p

    @property
    def seq_embedding(self) -> Tensor:
        return self.embedding_seq.embedding_element.weight.data

    @property
    def item_embedding(self) -> Tensor:
        return self.embedding_item.embedding_element.weight.data

    @property
    def seq_meta_embedding(self) -> Tensor:
        return self.embedding_seq.embedding_meta.weight.data

    @property
    def item_meta_embedding(self) -> Tensor:
        return self.embedding_item.embedding_meta.weight.data


class OldAttentiveModel(PyTorchModel):
    """OldAttentiveModel（古い提案モデル）のクラス"""

    def __init__(
        self,
        num_seq: int,
        num_item: int,
        num_seq_meta: int,
        num_item_meta: int,
        num_seq_meta_types: int,
        num_item_meta_types: int,
        sequences: List[List[int]],
        seq_meta_indices: Tensor,
        seq_meta_weights: Tensor,
        item_meta_indices: Tensor,
        item_meta_weights: Tensor,
        d_model: int = 128,
        init_embedding_std: float = 1,
        normalize_embedding_dim: bool = True,
        max_embedding_norm: Optional[float] = None,
        negative_sample_size: int = 30,
    ) -> None:
        """
        OldAttentiveModel（古い提案モデル）のクラスを生成する

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
        """
        super().__init__()
        self.d_model = d_model

        self.embedding_seq = MetaEmbeddingLayer(
            num_element=num_seq,
            num_meta=num_seq_meta,
            num_meta_types=num_seq_meta_types,
            d_model=d_model,
            meta_indices=seq_meta_indices,
            meta_weights=seq_meta_weights,
            normalize_embedding_dim=normalize_embedding_dim,
            max_embedding_norm=max_embedding_norm,
            init_embedding_std=init_embedding_std,
        )
        self.embedding_item = MetaEmbeddingLayer(
            num_element=num_item,
            num_meta=num_item_meta,
            num_meta_types=num_item_meta_types,
            d_model=d_model,
            meta_indices=item_meta_indices,
            meta_weights=item_meta_weights,
            normalize_embedding_dim=normalize_embedding_dim,
            max_embedding_norm=max_embedding_norm,
            init_embedding_std=init_embedding_std,
        )

        self.output = NegativeSampling(
            d_model=d_model,
            num_item=num_item,
            sequences=sequences,
            negative_sample_size=negative_sample_size,
            init_embedding_std=init_embedding_std,
        )

    def calc_out(
        self,
        seq_index: Tensor,
        item_indices: Tensor,
        target_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        c = self.calc_prediction_vector(
            seq_index=seq_index,
            item_indices=item_indices,
        )
        return self.output.forward(c, target_index)

    def calc_prediction_vector(
        self,
        seq_index: Tensor,
        item_indices: Tensor,
    ) -> Tensor:
        u = self.embedding_seq.forward(seq_index)
        V = self.embedding_item.forward(item_indices)

        Q = torch.reshape(u, (-1, 1, self.d_model))
        K = V
        V = V
        p = torch.reshape(attention(Q, K, V), (-1, self.d_model))

        return p

    @property
    def seq_embedding(self) -> Tensor:
        return self.embedding_seq.embedding_element.weight.data

    @property
    def item_embedding(self) -> Tensor:
        return self.embedding_item.embedding_element.weight.data

    @property
    def seq_meta_embedding(self) -> Tensor:
        return self.embedding_seq.embedding_meta.weight.data

    @property
    def item_meta_embedding(self) -> Tensor:
        return self.embedding_item.embedding_meta.weight.data

    @property
    def output_item_embedding(self) -> Tensor:
        return self.output.embedding.embedding.weight.data


class Doc2Vec(PyTorchModel):
    """Original Doc2Vec"""

    def __init__(
        self,
        num_seq: int,
        num_item: int,
        sequences: List[List[int]],
        d_model: int = 128,
        max_embedding_norm: Optional[float] = None,
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

        self.embedding_seq = NormalizedEmbeddingLayer(
            num_seq, d_model, max_norm=max_embedding_norm
        )
        self.embedding_item = NormalizedEmbeddingLayer(
            num_item, d_model, max_norm=max_embedding_norm
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
        item_indices: Tensor,
        target_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        c = self.calc_prediction_vector(
            seq_index=seq_index,
            item_indices=item_indices,
        )
        return self.output.forward(c, target_index)

    def calc_prediction_vector(
        self,
        seq_index: Tensor,
        item_indices: Tensor,
    ) -> Tensor:
        window_size = item_indices.size(1)

        u = self.embedding_seq.forward(seq_index)
        V = self.embedding_item.forward(item_indices)

        p = (u + V.sum(dim=1)) / (window_size + 1)
        return p

    @property
    def seq_embedding(self) -> Tensor:
        return self.embedding_seq.weight.data

    @property
    def item_embedding(self) -> Tensor:
        return self.embedding_item.weight.data

    @property
    def output_item_embedding(self) -> Tensor:
        return self.output.embedding.embedding.weight.data
