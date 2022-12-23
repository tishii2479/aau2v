import random
from typing import Any, Dict, List, Tuple

import pandas as pd


def generate_toydata(
    data_name: str = "toydata",
    user_count_per_segment: int = 1000,
    item_count_per_segment: int = 50,
    seq_lengths: List[int] = [50],
    genders: List[str] = ["M", "F"],
    ages: List[str] = ["20", "30", "40", "50", "60"],
    genres: List[str] = ["M", "E", "F"],
    years: List[str] = ["1960", "1970", "1980", "1990", "2000"],
    genre_ratio: Dict[str, Dict[str, float]] = {
        "M": {
            "M": 0.60,
            "E": 0.30,
            "F": 0.10,
        },
        "F": {
            "M": 0.10,
            "E": 0.30,
            "F": 0.60,
        },
    },
    year_ratio: Dict[str, Dict[str, float]] = {
        "20": {
            "1960": 0.10,
            "1970": 0.10,
            "1980": 0.10,
            "1990": 0.10,
            "2000": 0.60,
        },
        "30": {
            "1960": 0.10,
            "1970": 0.10,
            "1980": 0.10,
            "1990": 0.60,
            "2000": 0.10,
        },
        "40": {
            "1960": 0.10,
            "1970": 0.10,
            "1980": 0.60,
            "1990": 0.10,
            "2000": 0.10,
        },
        "50": {
            "1960": 0.10,
            "1970": 0.60,
            "1980": 0.10,
            "1990": 0.10,
            "2000": 0.10,
        },
        "60": {
            "1960": 0.60,
            "1970": 0.10,
            "1980": 0.10,
            "1990": 0.10,
            "2000": 0.10,
        },
    },
    test_length: int = 20,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rnd = random.Random(seed)

    raw_sequences: Dict[str, List[str]] = {}
    test_raw_sequences: Dict[str, List[str]] = {}
    users: Dict[str, Dict[str, Any]] = {}
    items: Dict[str, Dict[str, Any]] = {}

    def get_user_name(user_id: int, gender: str, age: str, seq_length: int) -> str:
        return f"u_{user_id}_{gender}_{age}_{seq_length}_{gender}{user_id % 5 + 1}"

    def get_item_name(item_id: int, genre: str, year: str) -> str:
        return f"v_{item_id}_{genre}_{year}"

    for gender in genders:
        for age in ages:
            for seq_length in seq_lengths:
                for user_id in range(user_count_per_segment):
                    user_name = get_user_name(user_id, gender, age, seq_length)

                    # user-metadata
                    users[user_name] = {
                        "gender": gender,
                        "age": age,
                    }

                    genre_weight = list(genre_ratio[gender].values())
                    year_weight = list(year_ratio[age].values())

                    # trains
                    genre_list = rnd.choices(
                        list(genre_ratio[gender].keys()), genre_weight, k=seq_length
                    )
                    year_list = rnd.choices(
                        list(year_ratio[age].keys()), year_weight, k=seq_length
                    )
                    item_id_list = sorted(
                        [
                            rnd.randint(0, item_count_per_segment - 1)
                            for _ in range(seq_length)
                        ]
                    )
                    sequences = list(
                        map(
                            lambda x: get_item_name(*x),
                            zip(item_id_list, genre_list, year_list),
                        )
                    )
                    raw_sequences[user_name] = sequences

                    # tests
                    genre_list = rnd.choices(
                        list(genre_ratio[gender].keys()), genre_weight, k=test_length
                    )
                    year_list = rnd.choices(
                        list(year_ratio[age].keys()), year_weight, k=test_length
                    )
                    item_id_list = sorted(
                        [
                            rnd.randint(0, item_count_per_segment - 1)
                            for _ in range(test_length)
                        ]
                    )
                    test_sequences = list(
                        map(
                            lambda x: get_item_name(*x),
                            zip(item_id_list, genre_list, year_list),
                        )
                    )
                    test_raw_sequences[user_name] = test_sequences

    for genre in genres:
        for year in years:
            for item_id in range(item_count_per_segment):
                item_name = get_item_name(item_id, genre, year)
                items[item_name] = {"genre": genre, "year": year}

    user_df = pd.DataFrame(users.values(), index=users.keys())
    item_df = pd.DataFrame(items.values(), index=items.keys())

    train_sequences = list(map(lambda s: " ".join(s), raw_sequences.values()))
    train_df = pd.DataFrame(
        train_sequences, index=raw_sequences.keys(), columns=["sequence"]
    )
    test_sequences = list(map(lambda s: " ".join(s), test_raw_sequences.values()))
    test_df = pd.DataFrame(
        test_sequences, index=test_raw_sequences.keys(), columns=["sequence"]
    )

    user_df.index.name = "user_id"
    item_df.index.name = "item_id"
    train_df.index.name = "user_id"
    test_df.index.name = "user_id"

    data_dir = f"../data/{data_name}/"

    user_df.to_csv(data_dir + "users.csv")
    item_df.to_csv(data_dir + "items.csv")
    train_df.to_csv(data_dir + "train.csv")
    test_df.to_csv(data_dir + "test.csv")

    return train_df, test_df, user_df, item_df
