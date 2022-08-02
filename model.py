from typing import List
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from math import sqrt
from sklearn.decomposition import PCA
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from gensim.models import word2vec

from toydata import create_toydata


def visualize_cluster(
    features: List[Tensor],
    num_cluster: int,
    cluster_labels: List[int],
    answer_labels: List[int]
):
    r'''
    Visualize cluster to 2d
    '''
    pca = PCA(n_components=2)
    pca.fit(features)
    pca_features = pca.fit_transform(features)

    colors = cm.rainbow(np.linspace(0, 1, num_cluster))

    plt.figure()

    for i in range(pca_features.shape[0]):
        plt.scatter(x=pca_features[i, 0], y=pca_features[i, 1],
                    color=colors[cluster_labels[i]], marker='$' + str(answer_labels[i]) + '$')

    plt.show()


def attention(Q: Tensor, K: Tensor, V: Tensor):
    a = F.softmax(torch.matmul(Q, K.mT) / sqrt(K.size(2)), dim=2)
    return torch.matmul(a, V)


class Model(nn.Module):
    def __init__(
        self,
        num_seq: int,
        num_item: int,
        d_model: int
    ):
        super().__init__()
        self.d_model = d_model

        self.W_seq = nn.Embedding(num_seq, d_model)
        self.W_item = nn.Embedding(num_item, d_model)

        self.W_seq_key = nn.Linear(d_model, d_model)
        self.W_seq_value = nn.Linear(d_model, d_model)

        self.W_item_key = nn.Linear(d_model, d_model)
        self.W_item_value = nn.Linear(d_model, d_model)

        self.projection = nn.Linear(d_model * 2, num_item)

    def forward(
        self,
        seq_index: Tensor,
        item_indicies: Tensor
    ):
        r'''
        seq_index:
            type: `int` or `long`
            shape: (batch_size, )
        item_indicies:
            type: `int` or `long`
            shape: (batch_size, seq_length)
        '''
        h_seq = self.W_seq.forward(seq_index)
        h_items = self.W_item.forward(item_indicies)

        Q = torch.reshape(self.W_seq_key(h_seq), (-1, 1, self.d_model))
        K = self.W_item_key(h_items)
        V = self.W_item_value(h_items)

        c = torch.reshape(attention(Q, K, V), (-1, self.d_model))
        c = torch.concat([c, self.W_seq_value(h_seq)], dim=1)

        v = self.projection.forward(c)

        return F.softmax(v, dim=1)

    @property
    def seq_embedding(self):
        return self.W_seq.weight

    @property
    def item_embedding(self):
        return self.W_item.weight


def to_sequential_data(
    sequences,
    length: int,
    item_le: LabelEncoder
):
    data = []
    for i, sequence in enumerate(sequences):
        for j in range(len(sequence) - length):
            seq_index = i
            item_indicies = item_le.transform(sequence[j:j + length])
            target_index = item_le.transform([sequence[j + length]])[0]
            data.append((seq_index, item_indicies, target_index))
    return data


class ToydataDataset(Dataset):
    def __init__(self, num_topic: int = 5, seq_length: int = 8, data_size: int = 20):
        (self.sequences, _), (self.items, _) = create_toydata(num_topic, data_size)
        self.item_le = LabelEncoder().fit(self.items)
        self.data = to_sequential_data(
            self.sequences, seq_length, self.item_le)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Trainer:
    def __init__(self, dataset: Dataset, num_seq: int, num_item: int, d_model: int, batch_size: int = 5, epochs: int = 100, lr: float = 0.00005):
        self.epochs = epochs
        self.model = Model(num_seq, num_item, d_model)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size=batch_size)

    def train(self):
        self.model.train()
        losses = []
        for epoch in range(self.epochs):
            total_loss = 0
            for data in self.data_loader:
                seq_index, item_indicies, target_index = data

                h = self.model.forward(seq_index, item_indicies)
                self.optimizer.zero_grad()
                loss = F.cross_entropy(h, target_index)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            total_loss /= len(self.data_loader)
            if epoch % 10 == 0:
                print(epoch, total_loss)

            losses.append(total_loss)

        return losses

    @torch.no_grad()
    def test(self):
        self.model.eval()
        for data in self.data_loader:
            seq_index, item_indicies, target_index = data
            h = self.model.forward(seq_index, item_indicies)
            print(h[:3], target_index[:3])
            break


def main():
    num_topic = 5
    seq_length = 8
    data_size = 20

    dataset = ToydataDataset(
        num_topic=num_topic, seq_length=seq_length, data_size=data_size)

    num_seq = len(dataset.sequences)
    num_item = len(dataset.items)

    d_model = 256
    batch_size = 256
    epochs = 200
    lr = 0.0005

    word2vec_model = word2vec.Word2Vec(dataset.sequences, vector_size=d_model)

    item_embeddings = torch.Tensor(
        [list(word2vec_model.wv[item]) for item in dataset.items])

    seq_embedding = []
    for sequence in dataset.sequences:
        b = [list(word2vec_model.wv[item])
             for item in sequence]
        a = torch.Tensor(b)
        seq_embedding.append(list(a.mean(dim=0)))

    seq_embedding = torch.Tensor(seq_embedding)

    trainer = Trainer(dataset, num_seq, num_item,
                      d_model, batch_size, epochs=epochs, lr=lr)

    trainer.model.item_embedding.data.copy_(item_embeddings)
    trainer.model.seq_embedding.data.copy_(seq_embedding)

    losses = trainer.train()
    trainer.test()

    torch.save(trainer.model.state_dict(), 'weights/model_1.pt')

    seq_embedding = trainer.model.seq_embedding

    kmeans = KMeans(n_clusters=num_topic)
    kmeans.fit(seq_embedding.detach().numpy())

    answer_labels = []
    for i in range(num_topic):
        answer_labels += [i] * data_size
    print(answer_labels)
    print(kmeans.labels_)

    visualize_cluster(seq_embedding.detach().numpy(),
                      num_topic, kmeans.labels_, answer_labels)

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
