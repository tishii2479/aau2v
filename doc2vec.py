import collections
from typing import Dict, List, Optional

import torch
from gensim.models import word2vec
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import SequenceDataset
from model import AttentiveModel
from util import top_cluster_items, visualize_cluster


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
        ignore_cache: bool = False,
        use_learnable_embedding: bool = False,
        verbose: bool = False
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
        '''
        self.epochs = epochs
        self.d_model = d_model
        self.model_path = model_path
        self.word2vec_path = word2vec_path
        self.ignore_cache = ignore_cache
        self.use_learnable_embedding = use_learnable_embedding
        self.verbose = verbose
        self.dataset = SequenceDataset(
            raw_sequences=raw_sequences, window_size=window_size)

        self.model = AttentiveModel(
            self.dataset.num_seq, self.dataset.num_item, d_model,
            self.dataset.sequences, negative_sample_size=negative_sample_size)

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size)

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
        for i, sequence in enumerate(raw_sequences):
            if self.verbose:
                print(i, len(raw_sequences))
            a = self.item_embeddings[self.dataset.item_le.transform(sequence)]
            seq_embedding_list.append(list(a.mean(dim=0)))

        seq_embedding = torch.Tensor(seq_embedding_list)
        self.model.seq_embedding.copy_(seq_embedding)
        self.model.seq_embedding.requires_grad = self.use_learnable_embedding

        print('learn_sequence_embedding end')

    def train(self, show_fig: bool = True) -> List[float]:
        self.learn_item_embedding(
            raw_sequences=self.dataset.raw_sequences, d_model=self.d_model,
            items=self.dataset.items)
        self.learn_sequence_embedding(raw_sequences=self.dataset.raw_sequences)

        self.model.load_state_dict(torch.load(self.model_path))  # type: ignore

        self.model.train()
        losses = []
        print('train start')
        for epoch in range(self.epochs):
            total_loss = 0.
            for i, data in enumerate(self.data_loader):
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

        torch.save(self.model.state_dict(), self.model_path)
        print(f'final loss: {losses[-1]}')

        if show_fig:
            plt.plot(losses)
            plt.show()

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
        top_item_infos = top_cluster_items(
            num_cluster=num_cluster, cluster_labels=cluster_labels,
            sequences=self.dataset.sequences,
            num_top_item=10, num_item=self.dataset.num_item)

        for cluster, (top_items, ratios) in enumerate(top_item_infos):
            print(f'Top items for cluster {cluster} (size {seq_cnt[cluster]}): \n')
            for index, item in enumerate(self.dataset.item_le.inverse_transform(top_items)):
                if item_name_dict is not None:
                    name = item_name_dict[item]
                else:
                    name = item
                print(name + ' ' + str(ratios[index]))
            print()

    def item_index(self, item: str | List[str]) -> int | List[int]:
        if type(item) == list:
            indicies: List[int] = self.dataset.item_le.transform(item)
            return indicies
        elif type(item) == str:
            item_index: int = self.dataset.item_le.transform([item])[0]
            return item_index
        else:
            raise TypeError

    def item_embedding(self, item: str | int) -> Tensor:
        if type(item) == str:
            return self.model.item_embedding[self.item_index(item)]
        elif type(item) == int:
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
