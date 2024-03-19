from collections import ChainMap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import gensim
import pandas as pd
from sklearn import datasets
from torchtext.data import get_tokenizer

from au2v.toydata import generate_toydata
from au2v.util import get_all_items


@dataclass
class RawDataset:
    """
    train_raw_sequences (Dict[str, List[str]])
        生の訓練用シーケンシャルデータ
        系列ID : [要素1, 要素2, ..., 要素n]
        例: "doc_001", [ "私", "は", "猫" ]
    item_metadata (Optional[Dict[str, Dict[str, Any]]], optional):
        要素の補助情報の辞書
        要素 : {
            補助情報ID: 補助情報の値
        }
        例: "私" : {
            "品詞": "名詞",
            "長さ": 1
        }
        Defaults to None.
    seq_metadata (Optional[Dict[str, Dict[str, Any]]], optional):
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
    """

    train_raw_sequences: Dict[str, List[str]]
    item_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    seq_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    test_raw_sequences_dict: Optional[Dict[str, Dict[str, List[str]]]] = None
    exclude_seq_metadata_columns: Optional[List[str]] = None
    exclude_item_metadata_columns: Optional[List[str]] = None


def load_raw_dataset(
    dataset_name: str,
    data_dir: str = "data/",
) -> RawDataset:
    # NOTE: hm, movielens, movielens-*は元データがないと動かない
    # TODO: いつか追加する
    match dataset_name:
        case "toydata-paper":
            # 卒論で使った人工データ
            data = generate_toydata(
                data_name="toydata-paper",
            )
            dataset = convert_toydata(*data)
        case "toydata-small":
            # テスト用の小さいデータ
            data = generate_toydata(
                data_name="toydata-small",
                user_count_per_segment=50,
                item_count_per_segment=3,
                seq_lengths=[20],
                test_length=20,
            )
            dataset = convert_toydata(*data)
        case "toydata-seq-lengths":
            # テスト用の小さいデータ
            data = generate_toydata(
                data_name="toydata-seq-lengths",
                user_count_per_segment=100,
                item_count_per_segment=10,
                seq_lengths=[25, 50, 75, 100],
                test_length=20,
            )
            dataset = convert_toydata(*data)
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
        case "movielens-new":
            dataset = create_movielens_data(
                train_path=f"{data_dir}/ml-1m-new/train.csv",
                test_paths={
                    "train-size=0": f"{data_dir}/ml-1m-new/test-0.csv",
                    "train-size=10": f"{data_dir}/ml-1m-new/test-10.csv",
                    "train-size=20": f"{data_dir}/ml-1m-new/test-20.csv",
                    "train-size=30": f"{data_dir}/ml-1m-new/test-30.csv",
                    "train-size=40": f"{data_dir}/ml-1m-new/test-40.csv",
                    "train-size=50": f"{data_dir}/ml-1m-new/test-50.csv",
                },
                user_path=f"{data_dir}/ml-1m-new/users.csv",
                movie_path=f"{data_dir}/ml-1m-new/movies.csv",
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

    return dataset


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
            sequences_df.index.values[
                max_data_size : max_data_size + test_data_size  # noqa
            ],
            sequences_df.sequence.values[
                max_data_size : max_data_size + test_data_size  # noqa
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


def convert_toydata(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
) -> RawDataset:
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
        index: [] if pd.isna(sequence) else sequence.split(" ")
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
