import collections
from typing import Dict, List, Optional, Tuple

import torch
import tqdm
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import ModelConfig, TrainerConfig
from data import Item, MetaData, Sequence, SequenceDataset
from models import AttentiveModel
from trainers import PyTorchTrainer
from util import (calc_cluster_occurence_array, calc_coherence,
                  calc_sequence_occurence_array, check_model_path,
                  top_cluster_items, visualize_cluster, visualize_loss)


class Analyst:
    def __init__(
        self,
        raw_sequences: List[Tuple[Sequence, MetaData]],
        items: List[Item],
        trainer_config: TrainerConfig,
        model_config: ModelConfig
    ):
        self.dataset = SequenceDataset(
            raw_sequences=raw_sequences, window_size=window_size)

        self.item_le = LabelEncoder().fit(item_metadata.keys())
        self.seq_le = LabelEncoder().fit(seq_metadata.keys())

        self.model = AttentiveModel(
            num_seq=self.dataset.num_seq, num_item=self.dataset.num_item, d_model=d_model,
            sequences=self.dataset.sequences, negative_sample_size=negative_sample_size)

        self.trainer = PyTorchTrainer()

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size)

        if load_model:
            self.model.load_state_dict(torch.load(self.model_path))  # type: ignore
        else:
            check_model_path(self.model_path)
            self.learn_item_embedding(
                raw_sequences=self.dataset.raw_sequences, d_model=self.d_model,
                items=self.dataset.items)
            self.learn_sequence_embedding(raw_sequences=self.dataset.raw_sequences)

    def learn_item_embedding(
        self,
        raw_sequences: List[List[str]],
        d_model: int,
        items: List[str]
    ) -> None:
        print('word2vec start.')
        word2vec_model = word2vec.Word2Vec(
            sentences=raw_sequences, vector_size=d_model, min_count=1)
        word2vec_model.save(self.word2vec_path)
        print('word2vec end.')

        item_embeddings = torch.Tensor(
            [list(word2vec_model.wv[item]) for item in items])
        self.model.item_embedding.copy_(item_embeddings)
        self.model.item_embedding.requires_grad = self.use_learnable_embedding

    def learn_sequence_embedding(
        self,
        raw_sequences: List[List[str]]
    ) -> None:
        print('learn_sequence_embedding start')

        # TODO: refactor
        seq_embedding_list = []
        for sequence in tqdm.tqdm(raw_sequences):
            a = self.item_embeddings[self.dataset.item_le.transform(sequence)]
            seq_embedding_list.append(list(a.mean(dim=0)))

        seq_embedding = torch.Tensor(seq_embedding_list)
        self.model.seq_embedding.copy_(seq_embedding)

        print('learn_sequence_embedding end')

    def train(self, show_fig: bool = True) -> List[float]:
        self.model.train()
        losses = []
        print('train start')
        for epoch in range(self.epochs):
            total_loss = 0.
            for i, data in enumerate(tqdm.tqdm(self.data_loader)):
                seq_index, item_indicies, target_index = data

                loss = self.model.forward(
                    seq_index, item_indicies, target_index)
                self.optimizer.zero_grad()
                loss.backward()  # type: ignore
                self.optimizer.step()

                if self.verbose:
                    print(i, len(self.data_loader), loss.item())
                total_loss += loss.item()

            total_loss /= len(self.data_loader)
            if epoch % 1 == 0:
                print(epoch, total_loss)

            losses.append(total_loss)
        print('train end')

        torch.save(self.model.state_dict(), self.model_path)

        if len(losses) > 0:
            print(f'final loss: {losses[-1]}')

        if show_fig and len(losses) > 0:
            visualize_loss(losses)

        return losses

    def cluster_sequences(self, num_cluster: int, show_fig: bool = True) -> List[int]:
        '''Cluster sequences using K-means

        Args:
            num_cluster (int): number of clusters
            show_fig (bool, optional): visualize cluster. Defaults to True.

        Returns:
            List[int]: cluster labels
        '''
        kmeans = KMeans(n_clusters=num_cluster)
        h_seq = self.seq_embeddings.detach().numpy()
        kmeans.fit(h_seq)
        cluster_labels: List[int] = kmeans.labels_

        if show_fig:
            visualize_cluster(h_seq, num_cluster, cluster_labels)

        return cluster_labels

    def top_items(
        self, num_cluster: int = 10, num_top_item: int = 10,
        item_name_dict: Optional[Dict[str, str]] = None,
        show_fig: bool = False
    ) -> None:
        cluster_labels = self.cluster_sequences(num_cluster, show_fig=show_fig)
        seq_cnt = collections.Counter(cluster_labels)
        cluster_occurence_array, cluster_size = calc_cluster_occurence_array(
            num_cluster=num_cluster, cluster_labels=cluster_labels,
            sequences=self.dataset.sequences, num_item=self.dataset.num_item)
        top_item_infos = top_cluster_items(
            num_cluster=num_cluster, cluster_occurence_array=cluster_occurence_array,
            cluster_size=cluster_size, num_top_item=num_top_item)

        for cluster, (top_items, ratios) in enumerate(top_item_infos):
            print(f'Top items for cluster {cluster} (size {seq_cnt[cluster]}): \n')
            for index, item in enumerate(self.dataset.item_le.inverse_transform(top_items)):
                if item_name_dict is not None:
                    name = item_name_dict[item]
                else:
                    name = item
                print(name + ' ' + str(ratios[index]))
            print()

    def calc_coherence(
        self, num_cluster: int = 10, num_top_item: int = 10,
        show_fig: bool = False
    ) -> float:
        '''Calculate coherence

        Args:
            num_cluster (int, optional): Number of clusters. Defaults to 10.
            num_top_item (int, optional): Number of top K items. Defaults to 10.
            show_fig (bool, optional): visualize. Defaults to False.

        Returns:
            float: coherence
        '''
        # TODO: refactor
        cluster_labels = self.cluster_sequences(num_cluster, show_fig=show_fig)
        cluster_occurence_array, cluster_size = calc_cluster_occurence_array(
            num_cluster=num_cluster, cluster_labels=cluster_labels,
            sequences=self.dataset.sequences, num_item=self.dataset.num_item)
        sequence_occurence_array = calc_sequence_occurence_array(
            sequences=self.dataset.sequences, num_item=self.dataset.num_item)
        top_item_infos = top_cluster_items(
            num_cluster=num_cluster, cluster_occurence_array=cluster_occurence_array,
            cluster_size=cluster_size, num_top_item=num_top_item)
        coherence = calc_coherence(
            sequence_occurence_array=sequence_occurence_array, top_item_infos=top_item_infos)
        print(f'coherence: {coherence}')
        return coherence

    @property
    def seq_embeddings(self) -> Tensor:
        return self.model.seq_embedding

    @property
    def item_embeddings(self) -> Tensor:
        return self.model.item_embedding
