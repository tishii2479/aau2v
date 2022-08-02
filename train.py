import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gensim.models import word2vec
import torch

from util import visualize_cluster
from data import SequenceDataset, create_toydata
from trainer import Trainer


def main():
    num_topic = 5
    window_size = 8
    data_size = 10

    d_model = 100
    batch_size = 64
    epochs = 30
    lr = 0.0005

    use_learnable_embedding = False

    dataset = SequenceDataset(
        data=create_toydata(num_topic, data_size), num_topic=num_topic, window_size=window_size, data_size=data_size)

    num_seq = len(dataset.sequences)
    num_item = len(dataset.items)

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

    trainer = Trainer(dataset=dataset, num_seq=num_seq, num_item=num_item,
                      d_model=d_model, batch_size=batch_size, epochs=epochs, lr=lr, model='model')

    trainer.model.item_embedding.data.copy_(item_embeddings)
    trainer.model.item_embedding.requires_grad = use_learnable_embedding
    trainer.model.seq_embedding.data.copy_(seq_embedding)
    trainer.model.seq_embedding.requires_grad = use_learnable_embedding

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
