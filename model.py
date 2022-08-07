import collections
from math import sqrt
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import tqdm
from gensim.models import word2vec
from sklearn.cluster import KMeans
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import SequenceDataset
from layer import NegativeSampling
from util import (calc_cluster_occurence_array, calc_coherence,
                  calc_sequence_occurence_array, check_model_path,
                  top_cluster_items, visualize_cluster, visualize_loss)


class AttentiveDoc2Vec:
    def __init__(
        self,
        raw_sequences: List[List[str]],
        model: str = 'attentive',
        d_model: int = 100,
        batch_size: int = 5,
        window_size: int = 8,
        epochs: int = 100,
        negative_sample_size: int = 10,
        lr: float = 0.00005,
        model_path: str = 'weights/model.pt',
        word2vec_path: str = 'weights/word2vec.model',
        use_learnable_embedding: bool = False,
        verbose: bool = False,
        load_model: bool = False
    ):
        '''Initialize Doc2Vec

        Args:
            raw_sequences (List[List[str]]):
                Raw representation of sequences (e.g. documents, purchase histories)
            model (str, optional):
                Model type to use. Defaults to 'attentive'.
            d_model (int, optional):
                Dimension size of the vector used through the model. Defaults to 100.
            batch_size (int, optional): Batch size. Defaults to 5.
            window_size (int, optional):
                Window size when predicting the next item. Defaults to 8.
            epochs (int, optional):
                Epoch number. Defaults to 100.
            lr (float, optional):
                Learning rate. Defaults to 0.00005.
            model_path (str, optional):
                Path of the weights being saved/loaded. Defaults to 'weights/model.pt'.
            word2vec_path (str, optional):
                Path of the weights of word2vec being saved/loaded.
                Defaults to 'weights/word2vec.model'.
            use_learnable_embedding (bool, optional):
                If true, tune embedding at training process. Defaults to False.
            verbose (bool, optional):
                Verbose when logging. Defaults to False.
            load_model (bool, optional):
                If true, load model from `model_path`. Defaults to False.
        '''
        self.epochs = epochs
        self.d_model = d_model
        self.model_path = model_path
        self.word2vec_path = word2vec_path
        self.use_learnable_embedding = use_learnable_embedding
        self.verbose = verbose
        self.dataset = SequenceDataset(
            raw_sequences=raw_sequences, window_size=window_size)

        self.model = AttentiveModel(
            num_seq=self.dataset.num_seq, num_item=self.dataset.num_item, d_model=d_model,
            sequences=self.dataset.sequences, negative_sample_size=negative_sample_size)

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

    @torch.no_grad()  # type: ignore
    def test(self) -> None:
        self.model.eval()
        for data in self.data_loader:
            seq_index, item_indicies, target_index = data
            loss = self.model.forward(seq_index, item_indicies, target_index)
            print(loss)
            break

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

    def item_index(self, item: str | List[str]) -> int | List[int]:
        if isinstance(item, list):
            indicies: List[int] = self.dataset.item_le.transform(item)
            return indicies
        elif isinstance(item, str):
            item_index: int = self.dataset.item_le.transform([item])[0]
            return item_index
        else:
            raise TypeError

    def item_embedding(self, item: str | int) -> Tensor:
        if isinstance(item, str):
            return self.model.item_embedding[self.item_index(item)]
        elif isinstance(item, int):
            # To avoid warning, wrap with str()
            return self.model.item_embedding[item]
        else:
            raise TypeError

    @property
    def seq_embeddings(self) -> Tensor:
        return self.model.seq_embedding

    @property
    def item_embeddings(self) -> Tensor:
        return self.model.item_embedding


class AttentiveModel(nn.Module):
    def __init__(
        self,
        num_seq: int,
        num_item: int,
        d_model: int,
        sequences: List[List[int]],
        concat: bool = False,
        negative_sample_size: int = 30
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.concat = concat

        self.W_seq = nn.Embedding(num_seq, d_model)
        self.W_item = nn.Embedding(num_item, d_model)

        self.W_item_key = nn.Linear(d_model, d_model)
        self.W_item_value = nn.Linear(d_model, d_model)

        output_dim = d_model * 2 if concat else d_model
        self.output = NegativeSampling(
            d_model=output_dim, num_item=num_item,
            sequences=sequences, negative_sample_size=negative_sample_size)

    def forward(
        self,
        seq_index: Tensor,
        item_indicies: Tensor,
        target_index: Tensor
    ) -> Tensor:
        r'''
        seq_index:
            type: `int` or `long`
            shape: (batch_size, )
        item_indicies:
            type: `int` or `long`
            shape: (batch_size, window_size)
        '''
        def attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
            a = F.softmax(torch.matmul(Q, K.mT) / sqrt(K.size(2)), dim=2)
            return torch.matmul(a, V)

        h_seq = self.W_seq.forward(seq_index)
        h_items = self.W_item.forward(item_indicies)

        Q = torch.reshape(h_seq, (-1, 1, self.d_model))
        K = self.W_item_key(h_items)
        V = self.W_item_value(h_items)

        c = torch.reshape(attention(Q, K, V), (-1, self.d_model))

        if self.concat:
            c = torch.concat([c, h_seq], dim=1)
        else:
            c += h_seq
        loss = self.output.forward(c, target_index)
        return loss

    @property
    def seq_embedding(self) -> Tensor:
        return self.W_seq.weight.data

    @property
    def item_embedding(self) -> Tensor:
        return self.W_item.weight.data


class OriginalDoc2Vec(nn.Module):
    def __init__(
        self,
        num_seq: int,
        num_item: int,
        d_model: int,
        sequences: List[List[int]]
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.W_seq = nn.Embedding(num_seq, d_model)
        self.W_item = nn.Embedding(num_item, d_model)

        self.projection = nn.Linear(d_model, num_item)

    def forward(
        self,
        seq_index: Tensor,
        item_indicies: Tensor,
        target_index: Tensor
    ) -> Tensor:
        r'''
        seq_index:
            type: `int` or `long`
            shape: (batch_size, )
        item_indicies:
            type: `int` or `long`
            shape: (batch_size, window_size)
        '''
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
