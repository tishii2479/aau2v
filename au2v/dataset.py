from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from au2v.toydata import generate_toydata


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
        case "movielens":
            dataset = create_movielens_data(
                train_path=f"{data_dir}/ml-1m/train.csv",
                test_paths={
                    "test": f"{data_dir}/ml-1m/test.csv",
                },
                user_path=f"{data_dir}/ml-1m/users.csv",
                movie_path=f"{data_dir}/ml-1m/movies.csv",
            )
        case _:
            raise ValueError(f"invalid dataset-name: {dataset_name}")

    return dataset


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
        index: sequence.split(" ")
        for index, sequence in zip(
            train_df.index.values,
            train_df.sequence.values,
        )
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
