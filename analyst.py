import collections
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans

from config import ModelConfig, TrainerConfig
from data import SequenceDataset
from trainers import PyTorchTrainer
from util import (
    calc_cluster_occurence_array,
    calc_coherence,
    calc_sequence_occurence_array,
    top_cluster_items,
    visualize_cluster,
    visualize_loss,
)


class Analyst:
    def __init__(
        self,
        dataset: SequenceDataset,
        trainer_config: TrainerConfig,
        model_config: ModelConfig,
    ):
        self.dataset = dataset
        self.trainer_config = trainer_config
        self.model_config = model_config

        self.trainer = PyTorchTrainer(
            dataset=self.dataset,
            trainer_config=trainer_config,
            model_config=model_config,
        )

    def fit(self, show_fig: bool = True) -> List[float]:
        losses = self.trainer.fit()
        if show_fig and len(losses) > 0:
            visualize_loss(losses)
        return losses

    def cluster_sequences(self, num_cluster: int, show_fig: bool = True) -> List[int]:
        """Cluster sequences using K-means

        Args:
            num_cluster (int): number of clusters
            show_fig (bool, optional): visualize cluster. Defaults to True.

        Returns:
            List[int]: cluster labels
        """
        kmeans = KMeans(n_clusters=num_cluster)
        h_seq = np.array(list(self.seq_embeddings.values()))
        kmeans.fit(h_seq)
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
        cluster_labels = self.cluster_sequences(num_cluster, show_fig=show_fig)
        seq_cnt = collections.Counter(cluster_labels)
        cluster_occurence_array, cluster_size = calc_cluster_occurence_array(
            num_cluster=num_cluster,
            cluster_labels=cluster_labels,
            sequences=self.dataset.sequences,
            num_item=self.dataset.num_item,
        )
        top_item_infos = top_cluster_items(
            num_cluster=num_cluster,
            cluster_occurence_array=cluster_occurence_array,
            cluster_size=cluster_size,
            num_top_item=num_top_item,
        )

        for cluster, (top_items, ratios) in enumerate(top_item_infos):
            print(f"Top items for cluster {cluster} (size {seq_cnt[cluster]}): \n")
            for index, item in enumerate(
                self.dataset.item_le.inverse_transform(top_items)
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
        cluster_labels = self.cluster_sequences(num_cluster, show_fig=show_fig)
        cluster_occurence_array, cluster_size = calc_cluster_occurence_array(
            num_cluster=num_cluster,
            cluster_labels=cluster_labels,
            sequences=self.dataset.sequences,
            num_item=self.dataset.num_item,
        )
        sequence_occurence_array = calc_sequence_occurence_array(
            sequences=self.dataset.sequences, num_item=self.dataset.num_item
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

    @property
    def seq_embeddings(self) -> Dict[str, np.ndarray]:
        return self.trainer.seq_embedding

    @property
    def item_embeddings(self) -> Dict[str, np.ndarray]:
        return self.trainer.item_embedding
