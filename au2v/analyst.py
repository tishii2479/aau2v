from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from au2v.dataset_manager import SequenceDatasetManager
from au2v.layer import attention_weight, cosine_similarity
from au2v.model import PyTorchModel
from au2v.util import to_full_meta_value, visualize_heatmap


def calc_similarity(a: Tensor, b: Tensor, method: str = "inner-product") -> Tensor:
    match method:
        case "attention":
            sim: Tensor = attention_weight(a, b)
        case "cos":
            sim = cosine_similarity(a, b)
        case "inner-product":
            sim = torch.matmul(a, b.T)
        case _:
            raise ValueError(f"Invalid method for calc_similarity: {method}")
    return sim.squeeze()


class Analyst:
    def __init__(
        self,
        model: PyTorchModel,
        dataset_manager: SequenceDatasetManager,
    ):
        self.model = model
        self.dataset_manager = dataset_manager

    def similarity_between_seq_and_item_meta(
        self,
        seq_index: int,  # TODO: accept str as seq_id
        item_meta_name: str,  # TODO: accept List[str]
        method: str = "inner-product",
    ) -> pd.DataFrame:
        """系列と要素の補助情報の類似度を計算する

        Args:
            seq_index (int): 系列
            item_meta_name (str): 要素の補助情報の名前
            method (str): 類似度の計算方法

        Returns:
            pd.DataFrame: 系列と要素の補助情報の類似度
        """
        item_meta_values = self.dataset_manager.item_meta_dict[item_meta_name]
        item_meta_names = [
            to_full_meta_value(item_meta_name, value) for value in item_meta_values
        ]
        item_meta_indices = self.dataset_manager.item_meta_le.transform(item_meta_names)
        e_seq = self.model.seq_embedding[seq_index]
        e_item_metas = self.model.item_meta_embedding[item_meta_indices]
        weights = calc_similarity(e_seq, e_item_metas, method)
        item_meta_weights = [
            (weight.item(), name) for weight, name in zip(weights, item_meta_names)
        ]
        result = sorted(item_meta_weights)[::-1]
        return pd.DataFrame(result, columns=["similarity", "item_meta"])

    def similarity_between_seq_and_item(
        self,
        seq_index: int,
        num_recent_items: int = 10,
        method: str = "inner-product",
    ) -> pd.DataFrame:
        """
        系列と要素の類似度を計算する

        Args:
            seq_index (int): 系列
            num_recent_items (int, optional): 参照する要素の個数. Defaults to 10.
            method (str, optional): 類似度の計算方法. Defaults to "inner-product".

        Returns:
            pd.DataFrame: 系列と要素の類似度
        """
        item_indices = self.dataset_manager.train_dataset.sequences[seq_index][
            -num_recent_items:
        ]
        e_seq = self.model.seq_embedding[seq_index]
        e_items = self.model.item_embedding[item_indices]
        weights = calc_similarity(e_seq, e_items, method)
        item_names = self.dataset_manager.item_le.inverse_transform(item_indices)
        item_weights = [
            (weight.item(), name) for weight, name in zip(weights, item_names)
        ]
        result = sorted(item_weights)[::-1]
        return pd.DataFrame(result, columns=["similarity", "item"])

    def similarity_between_seq_meta_and_item_meta(
        self,
        seq_meta_name: str,
        seq_meta_value: str,
        item_meta_name: str,
        method: str = "inner-product",
    ) -> pd.DataFrame:
        """
        系列の補助情報seq_meta_nameと要素の補助情報item_meta_nameの類似度を全て求める

        Args:
            seq_meta_name (str): 系列の補助情報の名前
            seq_meta_value (str): 系列の補助情報の値
            item_meta_name (str): 要素の補助情報の値
            method (str, optional): 類似度の計算方法. Defaults to "inner-product".

        Returns:
            pd.DataFrame: 要素の補助情報ごとの類似度
        """
        seq_meta = to_full_meta_value(seq_meta_name, seq_meta_value)
        seq_meta_index = self.dataset_manager.seq_meta_le.transform([seq_meta])
        item_meta_values = self.dataset_manager.item_meta_dict[item_meta_name]
        item_meta_names = [
            to_full_meta_value(item_meta_name, value) for value in item_meta_values
        ]
        item_meta_indices = self.dataset_manager.item_meta_le.transform(item_meta_names)
        e_seq_meta = self.model.seq_meta_embedding[seq_meta_index]
        e_item_metas = self.model.item_meta_embedding[item_meta_indices]
        weights = calc_similarity(e_seq_meta, e_item_metas, method)
        meta_weights = [
            (weight.item(), name) for weight, name in zip(weights, item_meta_names)
        ]
        result = sorted(meta_weights)[::-1]
        return pd.DataFrame(result, columns=["similarity", "item_meta"])

    def analyze_seq(
        self,
        seq_index: int,
        method: str = "inner-product",
    ) -> pd.DataFrame:
        """
        系列seq_index固有の埋め込み表現と補助情報の埋め込み表現の、要素の補助情報に対する類似度を全て求める

        Args:
            seq_index (int): 対象の系列の番号
            method (str): 類似度の求め方    Defaults to "inner-product"
            num_top_values (int): 使用する項目の数  Defaults to 5
            verbose (bool): 詳細を表示するかどうか  Defaults to True

        Returns:
            pd.DataFrame: 系列seq_indexの要素の補助情報に対する類似度
        """
        item_meta_indices = list(range(self.dataset_manager.num_item_meta))
        seq_id = self.dataset_manager.seq_le.inverse_transform([seq_index])[0]
        seq_meta_dict = self.dataset_manager.seq_metadata[seq_id]
        seq_meta_names = [
            to_full_meta_value(name, value) for name, value in seq_meta_dict.items()
        ]
        seq_meta_indices = self.dataset_manager.seq_meta_le.transform(seq_meta_names)

        e_seq = self.model.seq_embedding[seq_index]
        e_item_metas = self.model.item_meta_embedding[item_meta_indices]

        item_meta_names = self.dataset_manager.item_meta_le.classes_

        result = []

        weights = calc_similarity(e_seq, e_item_metas, method)
        result += [
            (weight.item(), seq_id, item_meta_name)
            for weight, item_meta_name in zip(weights, item_meta_names)
        ]

        for seq_meta_index, seq_meta_name in zip(seq_meta_indices, seq_meta_names):
            e_seq_meta = self.model.seq_meta_embedding[seq_meta_index]
            weights = calc_similarity(e_seq_meta, e_item_metas, method)
            result += [
                (weight.item(), seq_meta_name, item_meta_name)
                for weight, item_meta_name in zip(weights, item_meta_names)
            ]

        result = sorted(result)[::-1]
        return pd.DataFrame(result, columns=["similarity", "seq", "item"])

    def visualize_similarity_heatmap(
        self,
        seq_meta_names: Optional[List[str]] = None,
        item_meta_names: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (12, 8),
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """系列の補助情報と要素の補助情報の類似度を可視化したヒートマップを作成する

        Args:
            seq_meta_names (Optional[List[str]], optional):
            系列の補助情報. Defaults to None.
            item_meta_names (Optional[List[str]], optional):
            要素の補助情報. Defaults to None.
            figsize (Tuple[float, float], optional):
            作成する画像の大きさ. Defaults to (12, 8).

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: ヒートマップ
        """
        seq_meta = self.seq_meta_embedding
        item_meta = self.item_meta_embedding

        seq_meta_keys = (
            seq_meta_names
            if seq_meta_names is not None
            else self.dataset_manager.seq_meta_le.classes_
        )
        item_meta_keys = (
            item_meta_names
            if item_meta_names is not None
            else self.dataset_manager.item_meta_le.classes_
        )

        data = np.zeros((len(seq_meta_keys), len(item_meta_keys)))
        for i, seq_key in enumerate(seq_meta_keys):
            for j, item_key in enumerate(item_meta_keys):
                data[i][j] = torch.dot(seq_meta[seq_key], item_meta[item_key])

        # genre:ActionとかをActionにする
        # seq_meta_keys = list(map(lambda s: s[s.find(":") + 1 :], seq_meta_keys))
        # item_meta_keys = list(map(lambda s: s[s.find(":") + 1 :], item_meta_keys))

        return visualize_heatmap(data, seq_meta_keys, item_meta_keys, figsize)

    @property
    def seq_embedding(self) -> Dict[str, Tensor]:
        return {
            seq_name: e_seq
            for seq_name, e_seq in zip(
                self.dataset_manager.seq_le.classes_, self.model.seq_embedding
            )
        }

    @property
    def item_embedding(self) -> Dict[str, Tensor]:
        return {
            item_name: e_item
            for item_name, e_item in zip(
                self.dataset_manager.item_le.classes_, self.model.item_embedding
            )
        }

    @property
    def seq_meta_embedding(self) -> Dict[str, Tensor]:
        return {
            seq_meta_name: e_seq_meta
            for seq_meta_name, e_seq_meta in zip(
                self.dataset_manager.seq_meta_le.classes_,
                self.model.seq_meta_embedding,
            )
        }

    @property
    def item_meta_embedding(self) -> Dict[str, Tensor]:
        return {
            item_meta_name: e_item_meta
            for item_meta_name, e_item_meta in zip(
                self.dataset_manager.item_meta_le.classes_,
                self.model.item_meta_embedding,
            )
        }
