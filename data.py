from random import choice, randint
from typing import Dict, List, Tuple

import pandas as pd
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self, sequences: List[List[int]], window_size: int = 8
    ) -> None:
        self.sequences = sequences
        self.data = to_sequential_data(sequences, window_size)

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
    print("to_sequential data end")
    return data


def create_hm_data() -> Tuple[List[str], Dict[str, str]]:
    sequences = pd.read_csv("data/hm/purchase_history.csv")
    items = pd.read_csv("data/hm/items.csv", dtype={"article_id": str})

    raw_sequences = [
        sequence.split(" ") for sequence in sequences.sequence.values[:1000]
    ]

    item_names = items.name.values
    item_ids = items.article_id.values

    item_name_dict = {item_ids[i]: item_names[i] for i in range(len(item_ids))}

    return raw_sequences, item_name_dict
