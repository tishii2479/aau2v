import os
import pickle
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import gensim
import pandas as pd
from sklearn import datasets
from torchtext.data import get_tokenizer

from dataset_manager import SequenceDatasetManager
from util import get_all_items


@dataclass
class RawDataset:
    # TODO: write doc
    train_raw_sequences: Dict[str, List[str]]
    item_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    seq_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    test_raw_sequences_dict: Optional[Dict[str, Dict[str, List[str]]]] = None
    exclude_seq_metadata_columns: Optional[List[str]] = None
    exclude_item_metadata_columns: Optional[List[str]] = None


def load_dataset_manager(
    dataset_name: str,
    dataset_dir: str,
    load_dataset: bool,
    save_dataset: bool,
    window_size: int = 8,
    data_dir: str = "data/",
) -> SequenceDatasetManager:
    pickle_path = Path(dataset_dir).joinpath(f"{dataset_name}.pickle")
    if load_dataset and os.path.exists(pickle_path):
        print(f"load cached dataset_manager from: {pickle_path}")
        with open(pickle_path, "rb") as f:
            dataset_manager: SequenceDatasetManager = pickle.load(f)
        return dataset_manager

    print(f"dataset_manager does not exist at: {pickle_path}, create dataset")

    match dataset_name:
        case "toydata":
            dataset = create_toydata(
                train_path=f"{data_dir}/toydata/train.csv",
                test_path=f"{data_dir}/toydata/test.csv",
                user_path=f"{data_dir}/toydata/users.csv",
                item_path=f"{data_dir}/toydata/items.csv",
            )
        case "toydata-simple":
            dataset = create_toydata(
                train_path=f"{data_dir}/toydata-simple/train.csv",
                test_path=f"{data_dir}/toydata-simple/test.csv",
                user_path=f"{data_dir}/toydata-simple/users.csv",
                item_path=f"{data_dir}/toydata-simple/items.csv",
            )
        case "toydata-simple-mf":
            dataset = create_toydata(
                train_path=f"{data_dir}/toydata-simple-mf/train.csv",
                test_path=f"{data_dir}/toydata-simple-mf/test.csv",
                user_path=f"{data_dir}/toydata-simple-mf/users.csv",
                item_path=f"{data_dir}/toydata-simple-mf/items.csv",
            )
        case "toydata-hard":
            dataset = create_toydata(
                train_path=f"{data_dir}/toydata-hard/train.csv",
                test_path=f"{data_dir}/toydata-hard/test.csv",
                user_path=f"{data_dir}/toydata-hard/users.csv",
                item_path=f"{data_dir}/toydata-hard/items.csv",
            )
        case "toydata-paper":
            dataset = create_toydata(
                train_path=f"{data_dir}/toydata-paper/train.csv",
                test_path=f"{data_dir}/toydata-paper/test.csv",
                user_path=f"{data_dir}/toydata-paper/users.csv",
                item_path=f"{data_dir}/toydata-paper/items.csv",
            )
        case "toydata-small":
            dataset = create_toydata(
                train_path=f"{data_dir}/toydata-small/train.csv",
                test_path=f"{data_dir}/toydata-small/test.csv",
                user_path=f"{data_dir}/toydata-small/users.csv",
                item_path=f"{data_dir}/toydata-small/items.csv",
            )
        case "hm":
            dataset = create_hm_data(
                purchase_history_path=f"{data_dir}/hm/filtered_purchase_history.csv",
                item_path=f"{data_dir}/hm/items.csv",
                customer_path=f"{data_dir}/hm/customers.csv",
                max_data_size=1000,
                test_data_size=500,
            )
        case "movielens":
            dataset = create_movielens_data(
                train_path=f"{data_dir}/ml-1m/train.csv",
                test_paths={
                    "train-size=10": f"{data_dir}/ml-1m/test-10.csv",
                    "train-size=20": f"{data_dir}/ml-1m/test-20.csv",
                    "train-size=30": f"{data_dir}/ml-1m/test-30.csv",
                    "train-size=40": f"{data_dir}/ml-1m/test-40.csv",
                    "train-size=50": f"{data_dir}/ml-1m/test-50.csv",
                },
                user_path=f"{data_dir}/ml-1m/users.csv",
                movie_path=f"{data_dir}/ml-1m/movies.csv",
            )
        case "movielens-simple":
            dataset = create_movielens_data(
                train_path=f"{data_dir}/ml-1m/train.csv",
                test_paths={
                    "train-size=10": f"{data_dir}/ml-1m/test-10.csv",
                    "train-size=20": f"{data_dir}/ml-1m/test-20.csv",
                    "train-size=30": f"{data_dir}/ml-1m/test-30.csv",
                    "train-size=40": f"{data_dir}/ml-1m/test-40.csv",
                    "train-size=50": f"{data_dir}/ml-1m/test-50.csv",
                },
                user_path=f"{data_dir}/ml-1m/users.csv",
                movie_path=f"{data_dir}/ml-1m/movies.csv",
                user_columns=["gender"],
                movie_columns=["genre"],
            )
        case "movielens-equal-gender":
            dataset = create_movielens_data(
                train_path=f"{data_dir}/ml-1m/equal-gender-train.csv",
                test_paths={
                    "train-size=10": f"{data_dir}/ml-1m/test-10.csv",
                    "train-size=20": f"{data_dir}/ml-1m/test-20.csv",
                    "train-size=30": f"{data_dir}/ml-1m/test-30.csv",
                    "train-size=40": f"{data_dir}/ml-1m/test-40.csv",
                    "train-size=50": f"{data_dir}/ml-1m/test-50.csv",
                },
                user_path=f"{data_dir}/ml-1m/users.csv",
                movie_path=f"{data_dir}/ml-1m/movies.csv",
            )
        case "20newsgroup":
            dataset = create_20newsgroup_data(
                max_data_size=1000,
                min_seq_length=50,
                test_data_size=500,
            )
        case "20newsgroup-small":
            # テスト用の小さいデータ
            dataset = create_20newsgroup_data(
                max_data_size=10,
                min_seq_length=50,
                test_data_size=50,
            )
        case _:
            raise ValueError(f"invalid dataset-name: {dataset_name}")

    dataset_manager = SequenceDatasetManager(
        train_raw_sequences=dataset.train_raw_sequences,
        test_raw_sequences_dict=dataset.test_raw_sequences_dict,
        item_metadata=dataset.item_metadata,
        seq_metadata=dataset.seq_metadata,
        exclude_seq_metadata_columns=dataset.exclude_seq_metadata_columns,
        exclude_item_metadata_columns=dataset.exclude_item_metadata_columns,
        window_size=window_size,
    )

    if save_dataset:
        print(f"dumping dataset_manager to: {pickle_path}")
        with open(pickle_path, "wb") as f:
            pickle.dump(dataset_manager, f)
        print(f"dumped dataset_manager to: {pickle_path}")

    return dataset_manager


def create_hm_data(
    purchase_history_path: str,
    item_path: str,
    customer_path: str,
    max_data_size: int,
    test_data_size: int,
) -> RawDataset:
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

    return RawDataset(
        train_raw_sequences=raw_sequences,
        item_metadata=item_metadata,
        seq_metadata=customer_metadata,
        test_raw_sequences_dict=test_raw_sequences_dict,
        exclude_item_metadata_columns=["prod_name"],
    )


def create_toydata(
    train_path: str,
    test_path: str,
    user_path: str,
    item_path: str,
) -> RawDataset:
    train_df = pd.read_csv(train_path, dtype={"user_id": str}, index_col="user_id")
    test_df = pd.read_csv(test_path, dtype={"user_id": str}, index_col="user_id")
    user_df = pd.read_csv(user_path, dtype={"user_id": str}, index_col="user_id")
    item_df = pd.read_csv(item_path, dtype={"item_id": str}, index_col="item_id")
    train_raw_sequences = {
        user_name: sequence.split(" ")
        for user_name, sequence in zip(
            train_df.index.values,
            train_df.sequence.values,
        )
    }
    test_raw_sequences = {
        user_name: sequence.split(" ")
        for user_name, sequence in zip(
            test_df.index.values,
            test_df.sequence.values,
        )
    }
    test_raw_sequences_dict = {"test": test_raw_sequences}
    user_metadata = user_df.to_dict("index")
    item_metadata = item_df.to_dict("index")

    return RawDataset(
        train_raw_sequences=train_raw_sequences,
        item_metadata=item_metadata,
        seq_metadata=user_metadata,
        test_raw_sequences_dict=test_raw_sequences_dict,
    )


def create_movielens_data(
    train_path: str,
    test_paths: Dict[str, str],
    user_path: str,
    movie_path: str,
    user_columns: Optional[List[str]] = None,
    movie_columns: Optional[List[str]] = None,
) -> RawDataset:
    train_df = pd.read_csv(train_path, dtype={"user_id": str}, index_col="user_id")
    user_df = pd.read_csv(user_path, dtype={"user_id": str}, index_col="user_id")
    movie_df = pd.read_csv(movie_path, dtype={"movie_id": str}, index_col="movie_id")

    if user_columns is not None:
        user_df = user_df[user_columns]
    if movie_columns is not None:
        movie_df = movie_df[movie_columns]

    train_raw_sequences = {
        index: sequence.split(" ")
        for index, sequence in zip(train_df.index.values, train_df.sequence.values)
    }
    user_metadata = user_df.to_dict("index")
    movie_df.genre = movie_df.genre.apply(lambda s: s.split("|"))
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

    return RawDataset(
        train_raw_sequences=train_raw_sequences,
        item_metadata=movie_metadata,
        seq_metadata=user_metadata,
        test_raw_sequences_dict=test_raw_sequences_dict,
        exclude_item_metadata_columns=["title"],
    )


def create_20newsgroup_data(
    max_data_size: int = 1000,
    min_seq_length: int = 50,
    test_data_size: int = 500,
) -> RawDataset:
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

    return RawDataset(
        train_raw_sequences=train_raw_sequences,
        seq_metadata=seq_metadata,
        test_raw_sequences_dict=test_raw_sequences_dict,
    )
