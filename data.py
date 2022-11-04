import copy
from collections import ChainMap
from typing import Any, Dict, List, Optional, Set, Tuple

import gensim
import pandas as pd
import torch
import tqdm
from sklearn import datasets, preprocessing
from torch import Tensor
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer

from util import get_all_items, to_full_meta_value

MetaData = Dict[str, Any]
Item = Tuple[str, MetaData]
Sequence = List[Item]


class SequenceDatasetManager:
    def __init__(
        self,
        train_raw_sequences: Dict[str, List[str]],
        item_metadata: Optional[Dict[str, MetaData]] = None,
        seq_metadata: Optional[Dict[str, MetaData]] = None,
        test_raw_sequences_dict: Optional[Dict[str, Dict[str, List[str]]]] = None,
        exclude_seq_metadata_columns: Optional[List[str]] = None,
        exclude_item_metadata_columns: Optional[List[str]] = None,
        window_size: int = 8,
    ) -> None:
        """訓練データとテストデータを管理するクラス

        Args:
            train_raw_sequences (Dict[str, List[str]])
                生の訓練用シーケンシャルデータ
                系列ID : [要素1, 要素2, ..., 要素n]
                例: "doc_001", [ "私", "は", "猫" ]
            item_metadata (Optional[Dict[str, MetaData]], optional):
                要素の補助情報の辞書
                要素 : {
                    補助情報ID: 補助情報の値
                }
                例: "私" : {
                    "品詞": "名詞",
                    "長さ": 1
                }
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
            test_raw_sequences (Optional[Dict[str, Dict[str, List[str]]]], optional):
                生のテスト用シーケンシャルデータ
                複数のテストデータがあることを想定して、
                { テストデータ名 : テストデータ } の形式で管理
                テストデータの形式はtrain_raw_sequencesと一緒
                Defaults to None.
            exclude_seq_metadata_columns (Optional[List[str]], optional):
                `seq_metadata`の中で補助情報として扱わない列の名前のリスト（例: 顧客IDなど）
                Defaults to None.
            exclude_item_metadata_columns (Optional[List[str]], optional):
                `item_metadata`の中で補助情報として扱わない列の名前のリスト（例: 商品IDなど）
                Defaults to None.
            window_size (int, optional):
                学習するときに参照する過去の要素の個数
                Defaults to 8.
        """
        self.raw_sequences = copy.deepcopy(train_raw_sequences)
        if test_raw_sequences_dict is not None:
            # seq_idが一緒の訓練データとテストデータがあるかもしれないので、系列をマージする
            for _, test_raw_sequences in test_raw_sequences_dict.items():
                for seq_id, raw_sequence in test_raw_sequences.items():
                    if seq_id in self.raw_sequences:
                        self.raw_sequences[seq_id].extend(raw_sequence)
                    else:
                        self.raw_sequences[seq_id] = raw_sequence

        self.item_metadata = item_metadata if item_metadata is not None else {}
        self.seq_metadata = seq_metadata if seq_metadata is not None else {}

        items = get_all_items(self.raw_sequences)
        self.item_le = preprocessing.LabelEncoder().fit(items)
        self.item_meta_le, self.item_meta_dict = process_metadata(
            self.item_metadata, exclude_metadata_columns=exclude_item_metadata_columns
        )
        self.seq_le = preprocessing.LabelEncoder().fit(list(self.raw_sequences.keys()))
        self.seq_meta_le, self.seq_meta_dict = process_metadata(
            self.seq_metadata, exclude_metadata_columns=exclude_seq_metadata_columns
        )

        self.num_seq = len(self.raw_sequences)
        self.num_item = len(items)
        self.num_item_meta = len(self.item_meta_le.classes_)
        self.num_seq_meta = len(self.seq_meta_le.classes_)
        print(
            f"num_seq: {self.num_seq}, num_item: {self.num_item}, "
            + f"num_item_meta: {self.num_item_meta}, "
            + f"num_seq_meta: {self.num_seq_meta}"
        )

        self.train_dataset = SequenceDataset(
            raw_sequences=train_raw_sequences,
            item_metadata=self.item_metadata,
            seq_metadata=self.seq_metadata,
            seq_le=self.seq_le,
            item_le=self.item_le,
            seq_meta_le=self.seq_meta_le,
            item_meta_le=self.item_meta_le,
            window_size=window_size,
            exclude_seq_metadata_columns=exclude_seq_metadata_columns,
            exclude_item_metadata_columns=exclude_item_metadata_columns,
        )
        self.sequences = self.train_dataset.sequences

        if test_raw_sequences_dict is not None:
            self.test_dataset: Optional[Dict[str, SequenceDataset]] = {}
            for test_name, test_raw_sequences in test_raw_sequences_dict.items():
                self.test_dataset[test_name] = SequenceDataset(
                    raw_sequences=test_raw_sequences,
                    item_metadata=self.item_metadata,
                    seq_metadata=self.seq_metadata,
                    seq_le=self.seq_le,
                    item_le=self.item_le,
                    seq_meta_le=self.seq_meta_le,
                    item_meta_le=self.item_meta_le,
                    window_size=window_size,
                    exclude_seq_metadata_columns=exclude_seq_metadata_columns,
                    exclude_item_metadata_columns=exclude_item_metadata_columns,
                )
                self.sequences += self.test_dataset[test_name].sequences
        else:
            self.test_dataset = None


class SequenceDataset(Dataset):
    def __init__(
        self,
        raw_sequences: Dict[str, List[str]],
        item_metadata: Dict[str, MetaData],
        seq_metadata: Dict[str, MetaData],
        seq_le: preprocessing.LabelEncoder,
        item_le: preprocessing.LabelEncoder,
        seq_meta_le: preprocessing.LabelEncoder,
        item_meta_le: preprocessing.LabelEncoder,
        window_size: int = 8,
        exclude_seq_metadata_columns: Optional[List[str]] = None,
        exclude_item_metadata_columns: Optional[List[str]] = None,
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
            seq_metadata (Dict[str, MetaData]):
                系列の補助情報の辞書
                系列名: {
                    補助情報ID: 補助情報の値
                }
                例: "doc_1" : {
                    "ジャンル": "スポーツ",
                }
            seq_le (preprocessing.LabelEncoder):
                系列の名前とindexを対応づけるLabelEncoder
            item_le (preprocessing.LabelEncoder):
                要素の名前とindexを対応づけるLabelEncoder
            seq_meta_le (preprocessing.LabelEncoder):
                系列の補助情報の値とindexを対応づけるLabelEncoder
            item_meta_le (preprocessing.LabelEncoder):
                要素の補助情報の値とindexを対応づけるLabelEncoder
            window_size (int, optional):
                学習するときに参照する過去の要素の個数.
                Defaults to 8.
            exclude_metadata_columns (Optional[List[str]], optional):
                `item_metadata`の中で補助情報として扱わない列の名前のリスト（例: 単語IDなど）
                Defaults to None.
        """
        self.raw_sequences = raw_sequences

        self.sequences, self.data = to_sequential_data(
            raw_sequences=self.raw_sequences,
            item_metadata=item_metadata,
            seq_metadata=seq_metadata,
            seq_le=seq_le,
            item_le=item_le,
            seq_meta_le=seq_meta_le,
            item_meta_le=item_meta_le,
            window_size=window_size,
            exclude_seq_metadata_columns=exclude_seq_metadata_columns,
            exclude_item_metadata_columns=exclude_item_metadata_columns,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
        (
          seq_index,
          item_indicies,
          seq_meta_indicies,
          item_meta_indicies,
          target_index
        )
        """
        return self.data[idx]


def process_metadata(
    items: Dict[str, Dict[str, str]],
    exclude_metadata_columns: Optional[List[str]] = None,
) -> Tuple[preprocessing.LabelEncoder, Dict[str, Set[str]]]:
    """Process meta datas

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
    raw_sequences: Dict[str, List[str]],
    item_metadata: Dict[str, Dict[str, Any]],
    seq_metadata: Dict[str, Dict[str, Any]],
    seq_le: preprocessing.LabelEncoder,
    item_le: preprocessing.LabelEncoder,
    item_meta_le: preprocessing.LabelEncoder,
    seq_meta_le: preprocessing.LabelEncoder,
    window_size: int,
    exclude_seq_metadata_columns: Optional[List[str]] = None,
    exclude_item_metadata_columns: Optional[List[str]] = None,
) -> Tuple[List[List[int]], List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]]:
    """
    シーケンシャルデータを学習データに変換する

    Returns:
        Tuple[List[List[int]], List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]]:
            (sequences, data)
    """

    def get_item_meta_indicies(item_ids: List[int]) -> List[List[int]]:
        item_names = item_le.inverse_transform(item_ids)
        item_meta_indices: List[List[int]] = []
        for item_name in item_names:
            if item_name not in item_metadata:
                item_meta_indices.append([])
                continue
            item_meta: List[str] = []
            for meta_name, meta_value in item_metadata[item_name].items():
                if (
                    exclude_item_metadata_columns is not None
                    and meta_name in exclude_item_metadata_columns
                ):
                    continue
                item_meta.append(to_full_meta_value(meta_name, str(meta_value)))
            item_meta_indices.append(list(item_meta_le.transform(item_meta)))
        return item_meta_indices

    def get_seq_meta_indicies(seq_name: str) -> List[int]:
        seq_meta = []
        if seq_name not in seq_metadata:
            return []
        for meta_name, meta_value in seq_metadata[seq_name].items():
            seq_meta.append(to_full_meta_value(meta_name, meta_value))
        seq_meta_indicies: List[int] = seq_meta_le.transform(seq_meta)
        return seq_meta_indicies

    sequences = []
    data = []

    # TODO: parallelize
    print("to_sequential_data start")
    for seq_name, raw_sequence in tqdm.tqdm(raw_sequences.items()):
        sequence = item_le.transform(raw_sequence)
        sequences.append(sequence)

        seq_index = torch.tensor(seq_le.transform([seq_name])[0], dtype=torch.long)
        seq_meta_indicies = torch.tensor(
            get_seq_meta_indicies(seq_name),
            dtype=torch.long,
        )
        for j in range(len(sequence) - window_size):
            item_indicies = torch.tensor(
                sequence[j : j + window_size], dtype=torch.long
            )
            item_meta_indices = torch.tensor(
                get_item_meta_indicies(sequence[j : j + window_size]),
                dtype=torch.long,
            )
            target_index = torch.tensor(sequence[j + window_size], dtype=torch.long)
            data.append(
                (
                    seq_index,
                    item_indicies,
                    seq_meta_indicies,
                    item_meta_indices,
                    target_index,
                )
            )
    print("to_sequential_data end")
    return sequences, data


def create_hm_data(
    purchase_history_path: str = "data/hm/filtered_purchase_history.csv",
    item_path: str = "data/hm/items.csv",
    customer_path: str = "data/hm/customers.csv",
    max_data_size: int = 1000,
    test_data_size: int = 500,
) -> Tuple[
    Dict[str, List[str]],
    Optional[Dict[str, Dict[str, Any]]],
    Optional[Dict[str, Dict[str, Any]]],
    Optional[Dict[str, Dict[str, List[str]]]],
]:
    sequences_df = pd.read_csv(
        purchase_history_path, dtype={"customer_id": str}, index_col="customer_id"
    )
    items_df = pd.read_csv(item_path, dtype={"article_id": str}, index_col="article_id")
    customers_df = pd.read_csv(
        customer_path, dtype={"customer_id": str}, index_col="customer_id"
    )

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
    item_metadata = items_df.to_dict("index")
    customer_metadata = customers_df.to_dict("index")

    items_set = get_all_items(ChainMap(raw_sequences, test_raw_sequences))

    # item_set（raw_sequence, test_raw_sequence）に含まれている商品のみ抽出する
    item_metadata = dict(
        filter(lambda item: item[0] in items_set, item_metadata.items())
    )

    test_raw_sequences_dict = {"test": test_raw_sequences}

    return raw_sequences, item_metadata, customer_metadata, test_raw_sequences_dict


def create_movielens_data(
    train_path: str = "data/ml-1m/train.csv",
    test_paths: Dict[str, str] = {
        "train-size=10": "data/ml-1m/test-10.csv",
        "train-size=20": "data/ml-1m/test-20.csv",
        "train-size=30": "data/ml-1m/test-30.csv",
        "train-size=40": "data/ml-1m/test-40.csv",
        "train-size=50": "data/ml-1m/test-50.csv",
    },
    user_path: str = "data/ml-1m/users.csv",
    movie_path: str = "data/ml-1m/movies.csv",
    max_data_size: int = 1000,
    min_seq_length: int = 50,
    test_data_size: int = 500,
) -> Tuple[
    Dict[str, List[str]],
    Optional[Dict[str, Dict[str, Any]]],
    Optional[Dict[str, Dict[str, Any]]],
    Optional[Dict[str, Dict[str, List[str]]]],
]:
    train_df = pd.read_csv(train_path, dtype={"user_id": str}, index_col="user_id")
    user_df = pd.read_csv(user_path, dtype={"user_id": str}, index_col="user_id")
    movie_df = pd.read_csv(movie_path, dtype={"movie_id": str}, index_col="movie_id")

    train_raw_sequences = {
        index: sequence.split(" ")
        for index, sequence in zip(
            train_df.index.values,
            train_df.sequence.values,
        )
    }
    user_metadata = user_df.to_dict("index")
    movie_metadata = movie_df.to_dict("index")

    test_raw_sequences_dict: Dict[str, Dict[str, List[str]]] = {}

    for test_name, test_path in test_paths.items():
        test_df = pd.read_csv(test_path, dtype={"user_id": str}, index_col="user_id")
        test_raw_sequences_dict[test_name] = {
            index: sequence.split(" ")
            for index, sequence in zip(
                test_df.index.values,
                test_df.sequence.values,
            )
        }

    return train_raw_sequences, movie_metadata, user_metadata, test_raw_sequences_dict


def create_20newsgroup_data(
    max_data_size: int = 1000,
    min_seq_length: int = 50,
    test_data_size: int = 500,
) -> Tuple[
    Dict[str, List[str]],
    Optional[Dict[str, Dict[str, Any]]],
    Optional[Dict[str, Dict[str, Any]]],
    Optional[Dict[str, Dict[str, List[str]]]],
]:
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

    train_raw_sequences: Dict[str, List[str]] = {}
    test_raw_sequences: Dict[str, List[str]] = {}
    seq_metadata: Dict[str, Dict[str, Any]] = {}

    for doc_id, (target, document) in enumerate(
        zip(newsgroups_train.target, newsgroups_train.data)
    ):
        if (
            len(train_raw_sequences) + len(test_raw_sequences)
            >= max_data_size + test_data_size
        ):
            break

        tokens = tokenizer(document)
        sequence = []
        for word in tokens:
            if word in dictionary.token2id:
                sequence.append(word)

        if len(sequence) <= min_seq_length:
            continue

        seq_metadata[str(doc_id)] = {"target": target}

        if len(train_raw_sequences) < max_data_size:
            train_raw_sequences[str(doc_id)] = sequence
        else:
            test_raw_sequences[str(doc_id)] = sequence

    test_raw_sequences_dict: Dict[str, Dict[str, List[str]]] = {
        "test": test_raw_sequences
    }

    return train_raw_sequences, None, seq_metadata, test_raw_sequences_dict
