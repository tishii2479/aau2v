import datetime
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    data_dir = Path("data") / "ml-1m"
    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        engine="python",
        header=None,
        encoding="latin-1",
    )
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        encoding="latin-1",
    )
    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        encoding="latin-1",
    )

    users.columns = ["user_id", "gender", "age", "occupation", "zip"]
    movies.columns = ["movie_id", "title", "genre"]
    ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

    users = users.drop("zip", axis=1)
    ages = {
        1: "Under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+",
    }

    occupations = {
        0: "other",
        1: "academic/educator",
        2: "artist",
        3: "clerical/admin",
        4: "college/grad student",
        5: "customer service",
        6: "doctor/health care",
        7: "executive/managerial",
        8: "farmer",
        9: "homemaker",
        10: "K-12 student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales/marketing",
        15: "scientist",
        16: "self-employed",
        17: "technician/engineer",
        18: "tradesman/craftsman",
        19: "unemployed",
        20: "writer",
    }

    users.age = users.age.replace(ages)
    users.occupation = users.occupation.replace(occupations)
    movies["year"] = movies.title.str.extract("\((\d{4})\)", expand=True)  # noqa
    movies.year = movies.year.apply(lambda x: str(int(x) // 10 * 10))
    ratings = ratings.sort_values(by="timestamp")
    ratings.timestamp = ratings.timestamp.apply(datetime.datetime.fromtimestamp)
    split_date = datetime.datetime(year=2000, month=12, day=1)
    train_df = ratings[ratings.timestamp < split_date]
    test_df = ratings[ratings.timestamp >= split_date]
    train_raw_sequences = (
        train_df.groupby("user_id")
        .movie_id.agg(list)
        .apply(lambda ls: " ".join(map(str, ls)))
        .rename("sequence")
    )
    test_raw_sequences = (
        test_df.groupby("user_id")
        .movie_id.agg(list)
        .apply(lambda ls: " ".join(map(str, ls)))
        .rename("sequence")
    )

    train_raw_sequences.to_csv(data_dir / "train.csv")
    test_raw_sequences.to_csv(data_dir / "test.csv")
    movies.to_csv(data_dir / "movies.csv", index=False)
    users.to_csv(data_dir / "users.csv", index=False)
