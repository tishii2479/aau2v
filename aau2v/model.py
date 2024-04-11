import abc
import collections
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from aau2v.config import ModelConfig, TrainerConfig
from aau2v.dataset_center import SequenceDatasetCenter
from aau2v.layer import (
    EmbeddingLayer,
    MetaEmbeddingLayer,
    NegativeSampling,
    WeightSharedNegativeSampling,
    attention,
)


class PyTorchModel(nn.Module, metaclass=abc.ABCMeta):
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

    def output_rec_lists(
        self, seq_index: Tensor, item_indices: Tensor, cand_item_indices: Tensor, k: int
    ) -> list[list[int]]:
        h = self.calc_prediction_vector(seq_index=seq_index, item_indices=item_indices)
        e_v = self.embedding_item.forward(cand_item_indices)
        return torch.matmul(h, e_v.mT).argsort(dim=1, descending=True)[:, :k].tolist()

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

    @property
    def seq_embedding(self) -> Tensor:
        raise NotImplementedError()

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
    def out_item_embedding(self) -> Tensor:
        raise NotImplementedError()


class AttentiveAuxiliaryUser2Vec(PyTorchModel):
    """AttentiveAuxiliaryUser2Vec（AAU2V）のクラス"""

    def __init__(
        self,
        num_seq: int,
        num_item: int,
        num_seq_meta: int,
        num_item_meta: int,
        num_seq_meta_types: int,
        num_item_meta_types: int,
        item_counter: collections.Counter,
        seq_meta_indices: Tensor,
        seq_meta_weights: Tensor,
        item_meta_indices: Tensor,
        item_meta_weights: Tensor,
        device: str = "cpu",
        d_model: int = 128,
        init_embedding_std: float = 1,
        max_embedding_norm: Optional[float] = None,
        negative_sample_size: int = 5,
        use_weight_tying: bool = True,
        use_meta: bool = True,
        use_attention: bool = True,
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
        self.use_attention = use_attention

        if use_meta:
            self.embedding_seq: nn.Module = MetaEmbeddingLayer(
                num_element=num_seq,
                num_meta=num_seq_meta,
                num_meta_types=num_seq_meta_types,
                d_model=d_model,
                meta_indices=seq_meta_indices,
                meta_weights=seq_meta_weights,
                max_embedding_norm=max_embedding_norm,
                init_embedding_std=init_embedding_std,
            )
            self.embedding_item: nn.Module = MetaEmbeddingLayer(
                num_element=num_item,
                num_meta=num_item_meta,
                num_meta_types=num_item_meta_types,
                d_model=d_model,
                meta_indices=item_meta_indices,
                meta_weights=item_meta_weights,
                max_embedding_norm=max_embedding_norm,
                init_embedding_std=init_embedding_std,
            )
        else:
            self.embedding_seq = EmbeddingLayer(
                num_seq, d_model, max_norm=max_embedding_norm
            )
            self.embedding_item = EmbeddingLayer(
                num_item, d_model, max_norm=max_embedding_norm
            )

        if use_weight_tying:
            self.output: nn.Module = WeightSharedNegativeSampling(
                d_model=d_model,
                num_item=num_item,
                num_item_meta_types=num_item_meta_types,
                item_counter=item_counter,
                negative_sample_size=negative_sample_size,
                item_meta_indices=item_meta_indices,
                item_meta_weights=item_meta_weights,
                embedding_item=self.embedding_item,
                device=device,
            )
        else:
            self.output = NegativeSampling(
                d_model=d_model,
                num_item=num_item,
                item_counter=item_counter,
                device=device,
                init_embedding_std=init_embedding_std,
                negative_sample_size=negative_sample_size,
                max_embedding_norm=max_embedding_norm,
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

        if self.use_attention:
            Q = torch.reshape(u, (-1, 1, self.d_model))
            K = V
            V = V
            p = torch.reshape(attention(Q, K, V), (-1, self.d_model))
        else:
            # uとVの平均を予測ベクトルとする
            p = torch.cat([u.unsqueeze(1), V], dim=1).mean(dim=1)

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


class User2Vec(PyTorchModel):
    """Original User2Vec"""

    def __init__(
        self,
        num_seq: int,
        num_item: int,
        item_counter: collections.Counter,
        device: str = "cpu",
        d_model: int = 128,
        init_embedding_std: float = 1,
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

        self.embedding_seq = EmbeddingLayer(
            num_seq,
            d_model,
            max_norm=max_embedding_norm,
            std=init_embedding_std,
        )
        self.embedding_item = EmbeddingLayer(
            num_item,
            d_model,
            max_norm=max_embedding_norm,
            std=init_embedding_std,
        )

        self.output = NegativeSampling(
            d_model=d_model,
            num_item=num_item,
            item_counter=item_counter,
            device=device,
            init_embedding_std=init_embedding_std,
            negative_sample_size=negative_sample_size,
            max_embedding_norm=max_embedding_norm,
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


def load_model(
    dataset_center: SequenceDatasetCenter,
    trainer_config: TrainerConfig,
    model_config: ModelConfig,
) -> PyTorchModel:
    model: PyTorchModel
    match trainer_config.model_name:
        case "aau2v":
            model = AttentiveAuxiliaryUser2Vec(
                num_seq=dataset_center.num_seq,
                num_item=dataset_center.num_item,
                num_seq_meta=dataset_center.num_seq_meta,
                num_item_meta=dataset_center.num_item_meta,
                num_seq_meta_types=dataset_center.num_seq_meta_types,
                num_item_meta_types=dataset_center.num_item_meta_types,
                d_model=model_config.d_model,
                init_embedding_std=model_config.init_embedding_std,
                max_embedding_norm=model_config.max_embedding_norm,
                item_counter=dataset_center.item_counter,
                seq_meta_indices=dataset_center.seq_meta_indices.to(
                    trainer_config.device
                ),
                seq_meta_weights=dataset_center.seq_meta_weights.to(
                    trainer_config.device
                ),
                item_meta_indices=dataset_center.item_meta_indices.to(
                    trainer_config.device
                ),
                item_meta_weights=dataset_center.item_meta_weights.to(
                    trainer_config.device
                ),
                negative_sample_size=model_config.negative_sample_size,
                device=trainer_config.device,
                use_weight_tying=model_config.use_weight_tying,
                use_meta=model_config.use_meta,
                use_attention=model_config.use_attention,
            )
        case "user2vec":
            model = User2Vec(
                num_seq=dataset_center.num_seq,
                num_item=dataset_center.num_item,
                d_model=model_config.d_model,
                init_embedding_std=model_config.init_embedding_std,
                max_embedding_norm=model_config.max_embedding_norm,
                item_counter=dataset_center.item_counter,
                negative_sample_size=model_config.negative_sample_size,
                device=trainer_config.device,
            )
        case _:
            raise ValueError(f"invalid model_name: {trainer_config.model_name}")

    return model
