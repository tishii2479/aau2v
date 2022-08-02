from random import randint, choice, shuffle
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np


class SequenceDataset(Dataset):
    def __init__(self, data, num_topic: int = 5, window_size: int = 8, data_size: int = 20):
        (self.sequences, _), (self.items, _) = data
        self.item_le = LabelEncoder().fit(self.items)
        self.data = to_sequential_data(
            self.sequences, window_size, self.item_le)
        self.transformed_sequences = [self.item_le.transform(
            sequence) for sequence in self.sequences]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_toydata(num_topic: int, data_size: int):
    documents = []
    words = []
    key_words = [[] for _ in range(num_topic)]

    for _ in range(1, 201):
        s = ''
        for _ in range(10):
            s += chr(ord('a') + randint(0, 26))
        words.append(s)

    for i in range(num_topic):
        for j in range(1, 11):
            s = chr(ord('a') + i) * j
            key_words[i].append(s)

    for i in range(num_topic):
        for _ in range(data_size):
            doc = []
            for _ in range(randint(150, 200)):
                doc.append(choice(words))
                doc.append(choice(key_words[i]))
            documents.append(doc)

    for i in range(num_topic):
        words += key_words[i]

    word_embedding = torch.eye(len(words))

    return (documents, None), (words, word_embedding)


def create_labeled_toydata(num_topic: int):
    (documents, _), (words, word_embedding) = create_toydata(num_topic)
    labels = []
    for i in range(num_topic):
        for _ in range(5):
            labels.append(i)
    return (documents, labels), (words, word_embedding)


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


def create_hm_data():
    sequences = pd.read_csv('data/hm/purchase_history.csv')
    items = pd.read_csv('data/hm/items.csv', dtype={'article_id': str})

    raw_sequences = [sequence.split(' ')
                     for sequence in sequences.sequence.values[:1000]]
    seq_labels = None

    item_names = items.name.values
    item_ids = items.article_id.values

    item_list = [item_ids[i] for i in range(len(item_ids))]

    word2vec_model = Word2Vec.load('weights/word2vec.model')

    print(f'calculating item embedding, size: {len(items)}')
    item_embedding = torch.Tensor(
        np.array([word2vec_model.wv[id] for id in items.article_id.values]))
    print('calculating item embedding is done.')

    return (raw_sequences, seq_labels), (item_list, item_embedding)
