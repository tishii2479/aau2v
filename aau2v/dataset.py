from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from aau2v.toydata import generate_toydata


@dataclass
class RawDataset:
    """
    train_raw_sequences (dict[str, list[str]])
        Raw sequential data for training model
        format: { sequence : [item_1, item_2, ..., item_n] }
        example:
        {
            "user_001": [ "item_a", "item_b", "item_c" ],
            "user_002": [ "item_c", "item_d", "item_a" ],
            ...
        }
    item_metadata (Optional[dict[str, dict[str, Any]]], optional):
        Dictionary of item auxiliary information
        format: { item : { meta_name : meta_value_of_item }
        example:
        {
            "item_a": {
                "genre": "action",
                "year": 2020,
            },
            "item_b": {
                "genre": "comedy",
                "year": 2000,
            },
            ...
        }
        Defaults to None.
    seq_metadata (Optional[dict[str, dict[str, Any]]], optional):
        Dictionary of sequence (user) auxiliary information
        format: { sequence : { meta_name : meta_value_of_sequence }
        example:
        {
            "user_001": {
                "gender": "M",
                "age": 20,
            },
            "user_002": {
                "genre": "F",
                "age": 30,
            },
            ...
        }
        Defaults to None.
    test_raw_sequences (Optional[dict[str, dict[str, List[str]]]], optional):
        Raw sequential data for testing model
        To handle multiple test data, we manage data in dictionary format
        format: { test_name : test_raw_sequence }
        the format of `test_raw_sequence` is same as `train_raw_sequence`
        Defaults to None.
    exclude_seq_metadata_columns (Optional[List[str]], optional):
        List of columns which we do not use as auxiliary information in `seq_meta_data`
        example: ["user_id"]
        Defaults to None.
    exclude_item_metadata_columns (Optional[List[str]], optional):
        List of columns which we do not use as auxiliary information in `item_meta_data`
        example: ["item_id"]
        Defaults to None.
    """

    train_raw_sequences: dict[str, list[str]]
    test_raw_sequences_dict: dict[str, dict[str, list[str]]]
    item_metadata: Optional[dict[str, dict[str, Any]]] = None
    seq_metadata: Optional[dict[str, dict[str, Any]]] = None
    exclude_seq_metadata_columns: Optional[list[str]] = None
    exclude_item_metadata_columns: Optional[list[str]] = None


def load_raw_dataset(
    dataset_name: str,
    data_dir: str = "data",
) -> RawDataset:
    match dataset_name:
        case "toydata-paper":
            data = generate_toydata()
            dataset = convert_toydata(*data)
        case "toydata-small":
            data = generate_toydata(
                user_count_per_segment=50,
                item_count_per_segment=10,
                seq_lengths=[50],
                test_length=10,
            )
            dataset = convert_toydata(*data)
        case "movielens":
            ml_data_dir = Path(data_dir) / "ml-1m"
            dataset = create_movielens_data(
                train_path=ml_data_dir / "train.csv",
                test_paths={
                    "test": ml_data_dir / "test.csv",
                },
                user_path=ml_data_dir / "users.csv",
                movie_path=ml_data_dir / "movies.csv",
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
    train_path: Union[str, Path],
    test_paths: dict[str, Union[str, Path]],
    user_path: Union[str, Path],
    movie_path: Union[str, Path],
    user_columns: Optional[list[str]] = None,
    movie_columns: Optional[list[str]] = None,
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

    test_raw_sequences_dict: dict[str, dict[str, list[str]]] = {}

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
