from random import choice, randint
from typing import Any, Dict, List, Optional, Set, Tuple

import gensim
import pandas as pd
import torch
import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer

MetaData = Dict[str, Any]
Item = Tuple[str, MetaData]
Sequence = List[Item]


class SequenceDataset(Dataset):
    def __init__(
        self,
        raw_sequences: Dict[str, List[str]],
        item_metadata: Dict[str, MetaData],
        seq_metadata: Optional[Dict[str, MetaData]] = None,
        window_size: int = 8,
    ) -> None:
        self.seq_metadata = seq_metadata
        self.item_metadata = item_metadata
        self.raw_sequences = raw_sequences
        self.item_le = LabelEncoder().fit(item_metadata.keys())
        self.meta_le, self.meta_dict = process_metadata(item_metadata)

        print("transform sequence start")
        self.sequences = [
            self.item_le.transform(sequence)
            for sequence in tqdm.tqdm(self.raw_sequences)
        ]
        print("transform sequence end")

        self.num_seq = len(self.raw_sequences)
        self.num_item = len(self.item_metadata)
        self.num_meta = len(self.meta_le.classes_)

        print(
            f"num_seq: {self.num_seq}, num_item: {self.num_item}, "
            + f"num_meta: {self.num_meta}"
        )
        self.data = to_sequential_data(
            self.sequences,
            self.item_metadata,
            self.item_le,
            self.meta_le,
            window_size=window_size,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Returns:
        #   (seq_index, item_indicies, meta_indicies, target_index)
        return self.data[idx]


def process_metadata(
    items: Dict[str, Dict[str, str]]
) -> Tuple[LabelEncoder, Dict[str, Set[str]]]:
    """Process item meta datas

    Args:
        items (Dict[str, Dict[str, str]]):
            item data (item_id, (meta_name, meta_value))

    Returns:
        Tuple[LabelEncoder, Dict[str, Set[int]]]:
            (Label Encoder of meta data, Dictionary of list of meta datas)
    """
    meta_dict: Dict[str, Set[str]] = {}
    for _, meta_data in items.items():
        for meta_name, meta_value in meta_data.items():
            # FIXME: temporary, fix soon
            if meta_name == "prod_name":
                continue
            if meta_name not in meta_dict:
                meta_dict[meta_name] = set()
            meta_dict[meta_name].add(meta_value)

    all_meta_values: List[str] = []
    for meta_name, meta_values in meta_dict.items():
        for value in meta_values:
            # create str that is identical
            all_meta_values.append(meta_name + ":" + str(value))

    meta_le = LabelEncoder().fit(all_meta_values)

    return meta_le, meta_dict


def to_sequential_data(
    sequences: List[List[int]],
    items: Dict[str, Dict[str, str]],
    item_le: LabelEncoder,
    meta_le: LabelEncoder,
    window_size: int,
) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    def get_meta_indicies(item_ids: List[int]) -> List[List[int]]:
        item_names = item_le.inverse_transform(item_ids)
        meta_indices: List[List[int]] = []
        for item_name in item_names:
            item_meta: List[str] = []
            for meta_name, meta_value in items[item_name].items():
                # FIXME: temporary, fix soon
                if meta_name == "prod_name":
                    continue
                item_meta.append(meta_name + ":" + str(meta_value))
            meta_indices.append(list(meta_le.transform(item_meta)))
        return meta_indices

    data = []
    print("to_sequential_data start")
    for i, sequence in enumerate(tqdm.tqdm(sequences)):
        for j in range(len(sequence) - window_size):
            seq_index = torch.tensor(i, dtype=torch.long)
            item_indicies = torch.tensor(
                sequence[j : j + window_size], dtype=torch.long
            )
            target_index = torch.tensor(sequence[j + window_size], dtype=torch.long)
            meta_indices = torch.tensor(
                get_meta_indicies(sequence[j : j + window_size]),
                dtype=torch.long,
            )
            data.append((seq_index, item_indicies, meta_indices, target_index))
    print("to_sequential_data end")
    return data


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


def create_hm_data(
    purchase_history_path: str = "data/hm/filtered_purchase_history.csv",
    item_path: str = "data/hm/items.csv",
    max_data_size: int = 1000,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    sequences_df = pd.read_csv(purchase_history_path)
    items_df = pd.read_csv(item_path, dtype={"article_id": str}, index_col="article_id")

    raw_sequences = {
        index: sequence.split(" ")
        for index, sequence in zip(
            sequences_df.index.values[:max_data_size],
            sequences_df.sequence.values[:max_data_size],
        )
    }
    items = items_df.to_dict("index")

    items_set = set()
    for seq in raw_sequences:
        for item in seq:
            items_set.add(item)

    items = dict(filter(lambda item: item[0] in items_set, items.items()))

    return raw_sequences, items


def create_20newsgroup_data(
    max_data_size: int = 1000, min_seq_length: int = 50
) -> Tuple[List[List[str]], Optional[Dict[str, str]]]:
    newsgroups_train = fetch_20newsgroups(
        data_home="data",
        subset="train",
        remove=("headers", "footers", "quotes"),
        shuffle=False,
        random_state=0,
    )
    tokenizer = get_tokenizer("basic_english")

    dictionary = gensim.corpora.Dictionary(
        [tokenizer(document) for document in newsgroups_train.data]
    )
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
