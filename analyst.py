import collections
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from torch import Tensor

from dataset_manager import SequenceDatasetManager
from layer import attention_weight, cosine_similarity
from model import Model
from util import (
    calc_cluster_occurence_array,
    calc_coherence,
    calc_sequence_occurence_array,
    to_full_meta_value,
    top_cluster_items,
    visualize_cluster,
    visualize_vectors,
)


def calc_similarity(a: Tensor, b: Tensor, method: str = "inner-product") -> Tensor:
    match method:
        case "attention":
            sim = attention_weight(a, b)
        case "cos":
            sim = cosine_similarity(a, b)
        case "inner-product":
            sim = np.matmul(a, b.T)
    return sim.squeeze()


class Analyst:
    def __init__(
        self,
        model: Model,
        dataset_manager: SequenceDatasetManager,
    ):
        self.model = model
        self.dataset_manager = dataset_manager

    def cluster_embeddings(
        self, num_cluster: int, show_fig: bool = True, target: str = "sequence"
    ) -> List[int]:
        """Cluster using K-means

        Args:
            num_cluster (int): number of clusters
            show_fig (bool, optional): visualize cluster. Defaults to True.

        Returns:
            List[int]: cluster labels
        """
        kmeans = KMeans(n_clusters=num_cluster)
        match target:
            case "sequence":
                embeddings = self.model.seq_embedding.values()
            case "item":
                embeddings = self.model.item_embedding.values()
            case _:
                print(f"Invalid target: {target}")
                assert False

        h_seq = np.array(list(embeddings))
        print(f"Start k-means: {h_seq.shape}")
        kmeans.fit(h_seq)
        print("End k-means")
        cluster_labels: List[int] = kmeans.labels_

        if show_fig:
            visualize_cluster(list(h_seq), num_cluster, cluster_labels)

        return cluster_labels

    def top_items(
        self,
        num_cluster: int = 10,
        num_top_item: int = 10,
        item_name_dict: Optional[Dict[str, str]] = None,
        show_fig: bool = False,
    ) -> None:
        cluster_labels = self.cluster_embeddings(
            num_cluster, show_fig=show_fig, target="sequence"
        )
        seq_cnt = collections.Counter(cluster_labels)
        cluster_occurence_array, cluster_size = calc_cluster_occurence_array(
            num_cluster=num_cluster,
            cluster_labels=cluster_labels,
            sequences=self.dataset_manager.train_dataset.sequences,
            num_item=self.dataset_manager.num_item,
        )
        top_item_infos = top_cluster_items(
            num_cluster=num_cluster,
            cluster_occurence_array=cluster_occurence_array,
            cluster_size=cluster_size,
            num_top_item=num_top_item,
        )

        for cluster, (top_items, ratios) in enumerate(top_item_infos):
            print(f"Top items for cluster {cluster} (size {seq_cnt[cluster]}):")
            for index, item in enumerate(
                self.dataset_manager.item_le.inverse_transform(top_items)
            ):
                if item_name_dict is not None:
                    name = item_name_dict[item]
                else:
                    name = item
                print(name + " " + str(ratios[index]))
            print()

    def calc_coherence(
        self, num_cluster: int = 10, num_top_item: int = 10, show_fig: bool = False
    ) -> float:
        """Calculate coherence

        Args:
            num_cluster (int, optional): Number of clusters. Defaults to 10.
            num_top_item (int, optional): Number of top K items. Defaults to 10.
            show_fig (bool, optional): visualize. Defaults to False.

        Returns:
            float: coherence
        """
        # TODO: refactor
        cluster_labels = self.cluster_embeddings(
            num_cluster, show_fig=show_fig, target="sequence"
        )
        cluster_occurence_array, cluster_size = calc_cluster_occurence_array(
            num_cluster=num_cluster,
            cluster_labels=cluster_labels,
            sequences=self.dataset_manager.train_dataset.sequences,
            num_item=self.dataset_manager.num_item,
        )
        sequence_occurence_array = calc_sequence_occurence_array(
            sequences=self.dataset_manager.train_dataset.sequences,
            num_item=self.dataset_manager.num_item,
        )
        top_item_infos = top_cluster_items(
            num_cluster=num_cluster,
            cluster_occurence_array=cluster_occurence_array,
            cluster_size=cluster_size,
            num_top_item=num_top_item,
        )
        coherence = calc_coherence(
            sequence_occurence_array=sequence_occurence_array,
            top_item_infos=top_item_infos,
        )
        print(f"coherence: {coherence}")
        return coherence

    def similarity_between_seq_and_item_meta(
        self,
        seq_index: int,  # TODO: accept str as seq_id
        item_meta_name: str,  # TODO: accept List[str]
        num_top_values: int = 10,  # ISSUE: なくす?
        method: str = "inner-product",
        verbose: bool = True,
    ) -> List[Tuple[Tensor, str]]:
        # TODO: 戻り値をfloatにする
        item_meta_values = list(self.dataset_manager.item_meta_dict[item_meta_name])
        item_meta_names = [
            to_full_meta_value(item_meta_name, value) for value in item_meta_values
        ]
        item_meta_indicies = self.dataset_manager.item_meta_le.transform(
            item_meta_names
        )
        e_seq = self.model.seq_embedding[seq_index]
        e_item_metas = self.model.item_meta_embedding[item_meta_indicies]
        weights = calc_similarity(e_seq, e_item_metas, method)
        item_meta_weights = [
            (weight, name) for weight, name in zip(weights, item_meta_names)
        ]
        result = sorted(item_meta_weights)[::-1][:num_top_values]
        if verbose:
            print(f"similarity of seq: {seq_index} for meta: {item_meta_name}")
            for weight, name in result:
                print(f"{weight.item():.4f}", name)
        return result

    def similarity_between_seq_and_item(
        self,
        seq_index: int,
        num_recent_items: int = 10,
        method: str = "inner-product",
        verbose: bool = True,
    ) -> List[Tuple[Tensor, str]]:
        item_indicies = self.dataset_manager.train_dataset.sequences[seq_index][
            -num_recent_items:
        ]
        e_seq = self.model.seq_embedding[seq_index]
        e_items = self.model.item_embedding[item_indicies]
        weights = calc_similarity(e_seq, e_items, method)
        item_names = self.dataset_manager.item_le.inverse_transform(item_indicies)
        item_weights = [(weight, name) for weight, name in zip(weights, item_names)][
            :num_recent_items
        ]
        result = sorted(item_weights)[::-1]
        if verbose:
            print(f"item weights of seq: {seq_index}")
            for weight, name in result:
                print(
                    f"{weight.item():.4f}",
                    name,
                    self.dataset_manager.item_metadata[name],
                )
        return result

    def similarity_between_seq_meta_and_item_meta(
        self,
        seq_meta_name: str,
        seq_meta_value: str,
        item_meta_name: str,
        num_top_values: int = 10,
        method: str = "inner-product",
        verbose: bool = True,
    ) -> List[Tuple[Tensor, str]]:
        seq_meta = to_full_meta_value(seq_meta_name, seq_meta_value)
        seq_meta_index = self.dataset_manager.seq_meta_le.transform([seq_meta])
        item_meta_values = list(self.dataset_manager.item_meta_dict[item_meta_name])
        item_meta_names = [
            to_full_meta_value(item_meta_name, value) for value in item_meta_values
        ]
        item_meta_indicies = self.dataset_manager.item_meta_le.transform(
            item_meta_names
        )
        e_seq_meta = self.model.seq_meta_embedding[seq_meta_index]
        e_item_metas = self.model.item_meta_embedding[item_meta_indicies]
        weights = calc_similarity(e_seq_meta, e_item_metas, method)
        meta_weights = [
            (weight, name) for weight, name in zip(weights, item_meta_names)
        ]
        result = sorted(meta_weights)[::-1][:num_top_values]
        if verbose:
            print(f"similarity of seq meta: {seq_meta} for meta: {item_meta_name}")
            for weight, name in result:
                print(f"{weight.item():.4f}", name)
        return result

    def similar_items(
        self, item_index: int, num_items: int = 10
    ) -> List[Tuple[float, str]]:
        item_name = self.dataset_manager.item_le.inverse_transform([item_index])[0]

        item_embedding = self.model.item_embedding
        similar_items: List[Tuple[float, str]] = []

        h = item_embedding[item_name]

        item_names = self.dataset_manager.item_le.classes_

        for i, name in enumerate(item_names):
            if i == item_index:
                continue
            h_item = item_embedding[name]
            distance = np.sum((h - h_item) ** 2)
            similar_items.append((distance, name))

        similar_items.sort()

        print(f"similar_items of {item_name}")
        print(f"{item_name} Info: {self.dataset_manager.item_metadata[name]}")
        for distance, name in similar_items[:num_items]:
            item = self.dataset_manager.item_metadata[name]
            print(f"{name} Info: {item}, Distance: {distance}")

        return similar_items[:num_items]

    def similar_sequences(
        self, seq_index: int, num_seqs: int = 5
    ) -> List[Tuple[float, str]]:
        seq_name = self.dataset_manager.seq_le.inverse_transform([seq_index])[0]
        seq_embedding = self.model.seq_embedding

        similar_customers: List[Tuple[float, str]] = []

        h = seq_embedding[seq_name]

        seq_names = self.dataset_manager.seq_le.classes_

        for i, name in enumerate(seq_names):
            if i == seq_index:
                continue
            h_seq = seq_embedding[name]
            distance = np.sum((h - h_seq) ** 2)
            similar_customers.append((distance, name))

        similar_customers.sort()

        print(f"similar_items of {seq_name}")
        purchased_items = self.dataset_manager.raw_sequences[name]
        print("\n".join(purchased_items[-5:]))
        for distance, name in similar_customers[:num_seqs]:
            purchased_items = self.dataset_manager.raw_sequences[name]
            print(f"{name} Distance: {distance}")
            print("\n".join(purchased_items[-5:]))

        for distance, name in similar_customers[-num_seqs:]:
            purchased_items = self.dataset_manager.raw_sequences[name]
            print(f"{name} Distance: {distance}")
            print("\n".join(purchased_items[-5:]))

        return similar_customers[:num_seqs]

    def analyze_seq(
        self,
        seq_index: int,
        method: str = "inner-product",
        num_top_values: int = 5,
        verbose: bool = True,
    ) -> List[Tuple[Tensor, str]]:
        """
        系列seq_index固有の埋め込み表現と補助情報の埋め込み表現の、itemの補助情報に対する類似度を全て求める

        Args:
            seq_index (int): 対象の系列の番号
            method (str): 類似度の求め方    Defaults to "inner-product"
            num_top_values (int): 使用する項目の数  Defaults to 5
            verbose (bool): 詳細を表示するかどうか  Defaults to True
        """
        item_meta_indicies = list(range(self.dataset_manager.num_item_meta))
        seq_id = self.dataset_manager.seq_le.inverse_transform([seq_index])[0]
        seq_meta_dict = self.dataset_manager.seq_metadata[seq_id]
        seq_meta_names = [
            to_full_meta_value(name, value) for name, value in seq_meta_dict.items()
        ]
        seq_meta_indicies = self.dataset_manager.seq_meta_le.transform(seq_meta_names)

        e_seq = self.model.seq_embedding[seq_index]
        e_item_metas = self.model.item_meta_embedding[item_meta_indicies]

        item_meta_names = self.dataset_manager.item_le.classes_

        result: List[Tuple[Tensor, str]] = []

        weights = calc_similarity(e_seq, e_item_metas)
        result += [
            (weight, f"seq_id -> {item_meta_name}")
            for weight, item_meta_name in zip(weights, item_meta_names)
        ]

        for seq_meta_index, seq_meta_name in zip(seq_meta_indicies, seq_meta_names):
            e_seq_meta = self.model.seq_meta_embedding[seq_meta_index]
            weights = calc_similarity(e_seq_meta, e_item_metas)
            result += [
                (weight, f"{seq_meta_name} -> {item_meta_name}")
                for weight, item_meta_name in zip(weights, item_meta_names)
            ]

        result = sorted(result)[::-1][:num_top_values]
        if verbose:
            print(f"analysis of seq_id: {seq_id}")
            for weight, name in result:
                print(f"{weight.item():.4f}", name)
        return result

    def visualize_meta_embedding(
        self, seq_meta_name: str, item_meta_name: str, method: str = "pca"
    ) -> None:
        embeddings: Dict[str, np.ndarray] = {}
        for seq_meta_value in self.dataset_manager.seq_meta_dict[seq_meta_name]:
            full_seq_meta_value = to_full_meta_value(seq_meta_name, seq_meta_value)
            embeddings[full_seq_meta_value] = self.seq_meta_embedding[
                full_seq_meta_value
            ]
        for item_meta_value in self.dataset_manager.item_meta_dict[item_meta_name]:
            full_item_meta_value = to_full_meta_value(item_meta_name, item_meta_value)
            embeddings[full_item_meta_value] = self.item_meta_embedding[
                full_item_meta_value
            ]
        visualize_vectors(embeddings, method=method)

    @property
    def seq_embedding(self) -> Dict[str, np.ndarray]:
        return {
            seq_name: h_seq.detach().numpy()
            for seq_name, h_seq in zip(
                self.dataset_manager.seq_le.classes_, self.model.seq_embedding
            )
        }

    @property
    def item_embedding(self) -> Dict[str, np.ndarray]:
        return {
            item_name: h_item.detach().numpy()
            for item_name, h_item in zip(
                self.dataset_manager.item_le.classes_, self.model.item_embedding
            )
        }

    @property
    def seq_meta_embedding(self) -> Dict[str, np.ndarray]:
        return {
            seq_meta_name: h_seq_meta.detach().numpy()
            for seq_meta_name, h_seq_meta in zip(
                self.dataset_manager.seq_meta_le.classes_,
                self.model.seq_meta_embedding,
            )
        }

    @property
    def item_meta_embedding(self) -> Dict[str, np.ndarray]:
        return {
            item_meta_name: h_item_meta.detach().numpy()
            for item_meta_name, h_item_meta in zip(
                self.dataset_manager.item_meta_le.classes_,
                self.model.item_meta_embedding,
            )
        }
