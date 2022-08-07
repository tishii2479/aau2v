from itertools import chain
from random import choice, randint
from typing import Dict, List, Optional, Tuple

import gensim
import pandas as pd
import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer


class SequenceDataset(Dataset):
    def __init__(
        self,
        raw_sequences: List[List[str]],
        window_size: int = 8,
    ) -> None:
        self.raw_sequences = raw_sequences
        self.items = list(set(chain.from_iterable(self.raw_sequences)))
        self.item_le = LabelEncoder().fit(self.items)

        print('transform sequence start')
        self.sequences = [
            self.item_le.transform(sequence) for sequence in tqdm.tqdm(self.raw_sequences)]
        print('transform sequence end')

        self.num_seq = len(self.sequences)
        self.num_item = len(self.items)
        self.data = to_sequential_data(self.sequences, window_size)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, List[int], int]:
        # Returns:
        #   (seq_index, item_indicies, target_index)
        return self.data[idx]


def create_toydata(num_topic: int, data_size: int) -> List[List[str]]:
    documents = []
    words = []
    key_words: List[List[str]] = [[] for _ in range(num_topic)]

    for _ in range(1, 201):
        s = ""
        for _ in range(10):
            s += chr(ord("a") + randint(0, 26))
        words.append(s)

    for i in range(num_topic):
        for j in range(1, 11):
            s = chr(ord("a") + i) * j
            key_words[i].append(s)

    for i in range(num_topic):
        for _ in range(data_size):
            doc = []
            for _ in range(randint(150, 200)):
                doc.append(choice(words))
                doc.append(choice(key_words[i]))
            documents.append(doc)

    return documents


def create_labeled_toydata(
    num_topic: int, data_size: int
) -> Tuple[List[List[str]], List[int]]:
    documents = create_toydata(num_topic, data_size)
    labels = []
    for i in range(num_topic):
        for _ in range(data_size):
            labels.append(i)
    return documents, labels


def to_sequential_data(
    sequences: List[List[int]], length: int
) -> List[Tuple[int, List[int], int]]:
    data = []
    print("to_sequential_data start")
    for i, sequence in enumerate(sequences):
        for j in range(len(sequence) - length):
            seq_index = i
            item_indicies = sequence[j: j + length]
            target_index = sequence[j + length]
            data.append((seq_index, item_indicies, target_index))
    print("to_sequential_data end")
    return data


def create_hm_data(
    purchase_history_path: str = "data/hm/filtered_purchase_history.csv",
    item_path: str = "data/hm/items.csv",
) -> Tuple[List[List[str]], Dict[str, str]]:
    sequences = pd.read_csv(purchase_history_path)
    items = pd.read_csv(item_path, dtype={"article_id": str})

    raw_sequences = [sequence.split(" ") for sequence in sequences.sequence.values[:1000]]

    item_names = items.name.values
    item_ids = items.article_id.values

    item_name_dict = {
        item_ids[i]: item_names[i] + '(' + item_ids[i] + ')' for i in range(len(item_ids))}

    return raw_sequences, item_name_dict


def create_20newsgroup_data(
    max_data_size: int = 1000, min_seq_length: int = 50
) -> Tuple[List[List[str]], Optional[Dict[str, str]]]:
    newsgroups_train = fetch_20newsgroups(
        data_home='data', subset='train', remove=('headers', 'footers', 'quotes'),
        shuffle=False, random_state=0)
    tokenizer = get_tokenizer('basic_english')

    dictionary = gensim.corpora.Dictionary([
        tokenizer(document) for document in newsgroups_train.data])
    dictionary.filter_extremes(no_below=10, no_above=0.1)

    raw_sequences = []

    for document in newsgroups_train.data:
        tokens = tokenizer(document)
        sequence = []
        for word in tokens:
            if word in dictionary.token2id:
                sequence.append(word)
        if len(sequence) <= min_seq_length:
            continue
        raw_sequences.append(sequence)
        if len(raw_sequences) == max_data_size:
            break
    return raw_sequences, None
