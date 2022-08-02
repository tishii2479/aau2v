from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import KMeans

from data import create_toydata
from util import visualize_cluster


def main():
    num_topic = 5
    data_size = 10
    (documents, _), _ = create_toydata(num_topic=num_topic, data_size=data_size)

    # words：1文書ずつ、単語に分割したリスト
    # tags：文書識別用のタグ情報（リストで指定でき、複数のタグをつけることも可能）
    # for文1回あたりの入力イメージは[TaggedDocument([単語1,単語2,単語3,......],[文書タグ])]
    trainings = [TaggedDocument(words=documents[i], tags=[
                                str(i)]) for i in range(len(documents))]

    # 学習
    model = Doc2Vec(documents=trainings, dm=1, vector_size=100,
                    window=8, min_count=10, workers=4, epochs=100)

    # モデルの保存('hogehoge'というファイル名で保存)
    model.save('weights/doc2vec.model')

    # モデルのロード(既に学習済みのモデルがある場合)
    # model = Doc2Vec.load('../model/hogehoge.model')

    # 文書タグ0(1つ目の文書)と類似している文書を上から3件(topnで指定)取得し、文書タグと類似度のセットが返ってくる
    for tag in ['0', '10', '20', '30']:
        print(model.dv.most_similar(tag, topn=data_size))

    seq_embedding = [model.dv[str(i)] for i in range(len(documents))]

    kmeans = KMeans(n_clusters=num_topic)
    kmeans.fit(seq_embedding)

    answer_labels = []
    for i in range(num_topic):
        answer_labels += [i] * data_size
    print(answer_labels)
    print(kmeans.labels_)

    visualize_cluster(seq_embedding,
                      num_topic, kmeans.labels_, answer_labels)

# 結果イメージ[(タグ, 類似度), ......]
# [('10', 0.99978), ('2', 0.98553), ('8', 0.98123)]


if __name__ == '__main__':
    main()
