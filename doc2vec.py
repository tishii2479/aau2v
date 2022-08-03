import collections
from itertools import chain

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from data import create_hm_data
from util import top_cluster_items, visualize_cluster


def main() -> None:
    num_cluster = 10
    raw_sequences, item_name_dict = create_hm_data()
    items: set = set(chain.from_iterable(raw_sequences))
    item_le = LabelEncoder().fit(items)

    sequences = [item_le.transform(sequence)
                 for sequence in raw_sequences]

    trainings = [TaggedDocument(words=sequences[i], tags=[
                                str(i)]) for i in range(len(sequences))]

    model = Doc2Vec(documents=trainings, dm=1, vector_size=100,
                    window=8, min_count=10, workers=4, epochs=100)

    seq_embedding = [model.dv[str(i)] for i in range(len(sequences))]

    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(seq_embedding)

    cluster_labels = kmeans.labels_

    visualize_cluster(seq_embedding,
                      num_cluster, cluster_labels)

    seq_cnt = collections.Counter(cluster_labels)

    top_items = top_cluster_items(
        num_cluster, cluster_labels, sequences,
        num_top_item=10, num_item=len(items))

    for cluster, (top_items, ratios) in enumerate(top_items):
        print(f'Top items for cluster {cluster} (size {seq_cnt[cluster]}): \n')
        for index, item in enumerate(item_le.inverse_transform(top_items)):
            print(item_name_dict[item] + ' ' + str(ratios[index]))
        print()


if __name__ == '__main__':
    main()
