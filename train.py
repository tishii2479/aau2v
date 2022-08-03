import collections
from itertools import chain

import matplotlib.pyplot as plt
import torch
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from data import SequenceDataset, create_hm_data
from trainer import Trainer
from util import top_cluster_items, visualize_cluster


def main() -> None:
    num_cluster = 10
    window_size = 8

    d_model = 100
    batch_size = 64
    epochs = 10
    lr = 0.0005

    use_learnable_embedding = False

    raw_sequences, item_name_dict = create_hm_data()

    items: set = set(chain.from_iterable(raw_sequences))
    item_le = LabelEncoder().fit(list(items))
    print('transform sequence start')
    sequences = [item_le.transform(sequence)
                 for sequence in raw_sequences]
    print('transform sequence end')
    dataset = SequenceDataset(sequences=sequences, window_size=window_size)

    num_seq = len(dataset.sequences)
    num_item = len(items)

    print(num_seq, num_item)

    print('word2vec start.')
    word2vec_model = word2vec.Word2Vec(
        sentences=raw_sequences, vector_size=d_model, min_count=1)
    word2vec_model.save('weights/word2vec_hm.model')
    print('word2vec end.')

    item_embeddings = torch.Tensor(
        [list(word2vec_model.wv[item]) for item in items])

    # TODO: refactor
    seq_embedding_list = []
    for sequence in raw_sequences:
        b = [list(word2vec_model.wv[item])
             for item in sequence]
        a = torch.Tensor(b)
        seq_embedding_list.append(list(a.mean(dim=0)))

    seq_embedding: torch.Tensor = torch.Tensor(seq_embedding_list)

    trainer = Trainer(dataset=dataset, num_seq=num_seq, num_item=num_item,
                      d_model=d_model, batch_size=batch_size, epochs=epochs,
                      lr=lr, model='model')

    trainer.model.item_embedding.copy_(item_embeddings)
    trainer.model.item_embedding.requires_grad = use_learnable_embedding
    trainer.model.seq_embedding.copy_(seq_embedding)
    trainer.model.seq_embedding.requires_grad = use_learnable_embedding

    trainer.model.load_state_dict(
        torch.load('weights/model.pt'))  # type: ignore
    losses = trainer.train()
    trainer.test()

    torch.save(trainer.model.state_dict(), 'weights/model.pt')

    seq_embedding = trainer.model.seq_embedding

    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(seq_embedding.detach().numpy())

    cluster_labels = kmeans.labels_

    print(cluster_labels)

    visualize_cluster(seq_embedding.detach().numpy(),
                      num_cluster, cluster_labels)

    plt.plot(losses)
    plt.show()

    seq_cnt = collections.Counter(cluster_labels)

    top_item_infos = top_cluster_items(
        num_cluster, cluster_labels, sequences,
        num_top_item=10, num_item=num_item)

    for cluster, (top_items, ratios) in enumerate(top_item_infos):
        print(f'Top items for cluster {cluster} (size {seq_cnt[cluster]}): \n')
        for index, item in enumerate(item_le.inverse_transform(top_items)):
            print(item_name_dict[item] + ' ' + str(ratios[index]))
        print()
    print('loss:', losses[-1])


if __name__ == '__main__':
    main()
