import collections
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from data import create_hm_data
from util import visualize_cluster, top_cluster_items


def main():
    num_cluster = 10
    data_size = 10
    raw_sequences, item_name_dict = create_hm_data()
    items = set()
    for s in raw_sequences:
        for item in s:
            items.add(item)
    items = list(items)
    item_le = LabelEncoder().fit(items)

    sequences = [item_le.transform(sequence)
                 for sequence in raw_sequences]

    # words：1文書ずつ、単語に分割したリスト
    # tags：文書識別用のタグ情報（リストで指定でき、複数のタグをつけることも可能）
    # for文1回あたりの入力イメージは[TaggedDocument([単語1,単語2,単語3,......],[文書タグ])]
    trainings = [TaggedDocument(words=sequences[i], tags=[
                                str(i)]) for i in range(len(sequences))]

    # 学習
    model = Doc2Vec.load('weights/doc2vec.model')
    # model = Doc2Vec(documents=trainings, dm=1, vector_size=100,
    #                 window=8, min_count=10, workers=4, epochs=100)

    # モデルの保存('hogehoge'というファイル名で保存)
    # model.save('weights/doc2vec.model')

    # モデルのロード(既に学習済みのモデルがある場合)
    # model = Doc2Vec.load('../model/hogehoge.model')

    # 文書タグ0(1つ目の文書)と類似している文書を上から3件(topnで指定)取得し、文書タグと類似度のセットが返ってくる
    # for tag in ['0', '10', '20', '30']:
    #     print(model.dv.most_similar(tag, topn=data_size))

    seq_embedding = [model.dv[str(i)] for i in range(len(sequences))]

    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(seq_embedding)

    cluster_labels = kmeans.labels_

    visualize_cluster(seq_embedding,
                      num_cluster, cluster_labels)

    seq_cnt = collections.Counter(cluster_labels)

    top_items = top_cluster_items(
        num_cluster, cluster_labels, sequences, num_top_item=10, num_item=len(items))

    for cluster, (top_items, ratios) in enumerate(top_items):
        print(f'Top items for cluster {cluster} (size {seq_cnt[cluster]}): \n')
        for index, item in enumerate(item_le.inverse_transform(top_items)):
            print(item_name_dict[item] + ' ' + str(ratios[index]))
        print()
    # print('loss:', losses[-1])

# 結果イメージ[(タグ, 類似度), ......]
# [('10', 0.99978), ('2', 0.98553), ('8', 0.98123)]


if __name__ == '__main__':
    main()
