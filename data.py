from collections import ChainMap
from random import choice, randint
from typing import Any, Dict, List, Optional, Set, Tuple

import gensim
import pandas as pd
import torch
import tqdm
from sklearn import datasets, preprocessing
from torch import Tensor
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer

from util import to_full_meta_value

MetaData = Dict[str, Any]
Item = Tuple[str, MetaData]
Sequence = List[Item]


class SequenceDatasetManager:
    def __init__(
        self,
        train_raw_sequences: Dict[str, List[str]],
        item_metadata: Dict[str, MetaData],
        test_raw_sequences: Optional[Dict[str, List[str]]] = None,
        seq_metadata: Optional[Dict[str, MetaData]] = None,
        window_size: int = 8,
        exclude_metadata_columns: Optional[List[str]] = None,
    ) -> None:
        """
        訓練データとテストデータを管理するクラス

        Args:
            train_raw_sequences (Dict[str, List[str]])
                生の訓練用シーケンシャルデータ
                系列ID : [要素1, 要素2, ..., 要素n]
                例: "doc_001", [ "私", "は", "猫" ]
            item_metadata (Dict[str, MetaData]):
                要素の補助情報の辞書
                要素 : {
                    補助情報ID: 補助情報の値
                }
                例: "私" : {
                    "品詞": "名詞",
                    "長さ": 1
                }
            test_raw_sequences (Optional[Dict[str, List[str]]], optional):
                生のテスト用シーケンシャルデータ
                形式はtrain_raw_sequencesと一緒
                Defaults to None.
            seq_metadata (Optional[Dict[str, MetaData]], optional):
                系列の補助情報の辞書
                系列: {
                    補助情報ID: 補助情報の値
                }
                例: "doc_001" : {
                    "ジャンル": 動物,
                    "単語数": 3
                }
                Defaults to None.
            window_size (int, optional):
                学習するときに参照する過去の要素の個数.
                Defaults to 8.
            exclude_metadata_columns (Optional[List[str]], optional):
                `item_metadata`の中で補助情報として扱わない列の名前のリスト（例: 単語IDなど）
                Defaults to None.
        """
        self.item_le = preprocessing.LabelEncoder().fit(list(item_metadata.keys()))
        self.meta_le, self.meta_dict = process_metadata(
            item_metadata, exclude_metadata_columns=exclude_metadata_columns
        )
        self.item_metadata = item_metadata

        self.num_seq = len(train_raw_sequences)
        if test_raw_sequences is not None:
            self.num_seq += len(test_raw_sequences)
        self.num_item = len(item_metadata)
        self.num_meta = len(self.meta_le.classes_)

        print(
            f"num_seq: {self.num_seq}, num_item: {self.num_item}, "
            + f"num_meta: {self.num_meta}"
        )

        self.train_dataset = SequenceDataset(
            train_raw_sequences,
            item_metadata,
            self.item_le,
            self.meta_le,
            seq_metadata,
            window_size,
            exclude_metadata_columns,
        )
        if test_raw_sequences is not None:
            self.test_dataset: Optional[SequenceDataset] = SequenceDataset(
                test_raw_sequences,
                item_metadata,
                self.item_le,
                self.meta_le,
                seq_metadata,
                window_size,
                exclude_metadata_columns,
            )
        else:
            self.test_dataset = None

        if test_raw_sequences is not None:
            self.raw_sequences = ChainMap(train_raw_sequences, test_raw_sequences)
        else:
            self.raw_sequences = ChainMap(train_raw_sequences)

        self.sequences = self.train_dataset.sequences
        if self.test_dataset is not None:
            self.sequences += self.test_dataset.sequences

    @property
    def has_test_dataset(self) -> bool:
        return self.test_dataset is not None


class SequenceDataset(Dataset):
    def __init__(
        self,
        raw_sequences: Dict[str, List[str]],
        item_metadata: Dict[str, MetaData],
        item_le: preprocessing.LabelEncoder,
        meta_le: preprocessing.LabelEncoder,
        seq_metadata: Optional[Dict[str, MetaData]] = None,
        window_size: int = 8,
        exclude_metadata_columns: Optional[List[str]] = None,
    ) -> None:
        """
        補助情報を含んだシーケンシャルのデータを保持するクラス

        Args:
            raw_sequences (Dict[str, List[str]]):
                生のシーケンシャルデータ
                系列ID : [要素1, 要素2, ..., 要素n]
                例: "doc_001", [ "私", "は", "猫" ]
            item_metadata (Dict[str, MetaData]):
                要素の補助情報の辞書
                要素 : {
                    補助情報ID: 補助情報の値
                }
                例: "私" : {
                    "品詞": "名詞",
                    "長さ": 1
                }
            seq_metadata (Optional[Dict[str, MetaData]], optional):
                系列の補助情報の辞書
                系列: {
                    補助情報ID: 補助情報の値
                }
                例: "doc_001" : {
                    "ジャンル": 動物,
                    "単語数": 3
                }
            window_size (int, optional):
                学習するときに参照する過去の要素の個数.
                Defaults to 8.
            exclude_metadata_columns (Optional[List[str]], optional):
                `item_metadata`の中で補助情報として扱わない列の名前のリスト（例: 単語IDなど）
                Defaults to None.
        """
        self.seq_metadata = seq_metadata
        self.raw_sequences = raw_sequences

        print("transform sequence start")
        self.sequences = [
            item_le.transform(sequence)
            for sequence in tqdm.tqdm(self.raw_sequences.values())
        ]
        print("transform sequence end")

        self.data = to_sequential_data(
            self.sequences,
            item_metadata,
            item_le,
            meta_le,
            window_size=window_size,
            exclude_metadata_columns=exclude_metadata_columns,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Returns:
        #   (seq_index, item_indicies, meta_indicies, target_index)
        return self.data[idx]


def process_metadata(
    items: Dict[str, Dict[str, str]],
    exclude_metadata_columns: Optional[List[str]] = None,
) -> Tuple[preprocessing.LabelEncoder, Dict[str, Set[str]]]:
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
            if (
                exclude_metadata_columns is not None
                and meta_name in exclude_metadata_columns
            ):
                continue
            if meta_name not in meta_dict:
                meta_dict[meta_name] = set()
            meta_dict[meta_name].add(meta_value)

    all_meta_values: List[str] = []
    for meta_name, meta_values in meta_dict.items():
        for value in meta_values:
            # create str that is identical
            all_meta_values.append(to_full_meta_value(meta_name, value))

    meta_le = preprocessing.LabelEncoder().fit(all_meta_values)

    return meta_le, meta_dict


def to_sequential_data(
    sequences: List[List[int]],
    items: Dict[str, Dict[str, str]],
    item_le: preprocessing.LabelEncoder,
    meta_le: preprocessing.LabelEncoder,
    window_size: int,
    exclude_metadata_columns: Optional[List[str]] = None,
) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    def get_meta_indicies(item_ids: List[int]) -> List[List[int]]:
        item_names = item_le.inverse_transform(item_ids)
        meta_indices: List[List[int]] = []
        for item_name in item_names:
            item_meta: List[str] = []
            for meta_name, meta_value in items[item_name].items():
                if (
                    exclude_metadata_columns is not None
                    and meta_name in exclude_metadata_columns
                ):
                    continue
                item_meta.append(to_full_meta_value(meta_name, meta_value))
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
    test_data_size: int = 500,
) -> Tuple[
    Dict[str, List[str]], Dict[str, Dict[str, str]], Optional[Dict[str, List[str]]]
]:
    sequences_df = pd.read_csv(purchase_history_path)
    items_df = pd.read_csv(item_path, dtype={"article_id": str}, index_col="article_id")

    raw_sequences = {
        index: sequence.split(" ")
        for index, sequence in zip(
            sequences_df.index.values[:max_data_size],
            sequences_df.sequence.values[:max_data_size],
        )
    }
    test_raw_sequences = {
        index: sequence.split(" ")
        for index, sequence in zip(
            sequences_df.index.values[max_data_size : max_data_size + test_data_size],
            sequences_df.sequence.values[
                max_data_size : max_data_size + test_data_size
            ],
        )
    }
    items = items_df.to_dict("index")

    items_set = set()
    for seq in raw_sequences.values():
        for item in seq:
            items_set.add(item)
    for seq in test_raw_sequences.values():
        for item in seq:
            items_set.add(item)

    # item_set（raw_sequence, test_raw_sequence）に含まれている商品のみ抽出する
    items = dict(filter(lambda item: item[0] in items_set, items.items()))

    return raw_sequences, items, test_raw_sequences


def create_20newsgroup_data(
    max_data_size: int = 1000, min_seq_length: int = 50
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    newsgroups_train = datasets.fetch_20newsgroups(
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

    raw_sequences = {}
    item_metadata: Dict[str, Dict[str, str]] = {}

    for doc_id, document in enumerate(newsgroups_train.data):
        tokens = tokenizer(document)
        sequence = []
        for word in tokens:
            if word in dictionary.token2id:
                sequence.append(word)
                item_metadata[word] = {}
        if len(sequence) <= min_seq_length:
            continue
        raw_sequences[str(doc_id)] = sequence
        if len(raw_sequences) == max_data_size:
            break
    return raw_sequences, item_metadata
