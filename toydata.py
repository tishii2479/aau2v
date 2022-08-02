from random import randint, choice, shuffle
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset


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


class ToydataDataset(Dataset):
    def __init__(self, num_topic: int = 5, window_size: int = 8, data_size: int = 20):
        (self.sequences, _), (self.items, _) = create_toydata(num_topic, data_size)
        self.item_le = LabelEncoder().fit(self.items)
        self.data = to_sequential_data(
            self.sequences, window_size, self.item_le)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
