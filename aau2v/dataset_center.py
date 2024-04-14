import collections
import copy
from typing import Any, MutableSet, Optional

import torch
from sklearn import preprocessing
from torch import Tensor
from torch.utils.data import Dataset

from aau2v.dataset import RawDataset, load_raw_dataset


def to_full_meta_value(meta_name: str, meta_value: Any) -> str:
    """
    Generate identical string that describes the meta value

    Args:
        meta_name (str): meta data name (column name)
        meta_value (Any): meta data value

    Returns:
        str: identical string that describes the meta value
    """
    return meta_name + ":" + str(meta_value)


class SequenceDatasetCenter:
    """
    訓練データとテストデータを管理するクラス
    """

    def __init__(self, dataset: RawDataset, window_size: int) -> None:
        self.raw_sequences = copy.deepcopy(dataset.train_raw_sequences)
        self.item_metadata = (
            dataset.item_metadata if dataset.item_metadata is not None else {}
        )
        self.seq_metadata = (
            dataset.seq_metadata if dataset.seq_metadata is not None else {}
        )

        self.seq_le, self.item_le = get_seq_item_le(
            raw_sequences=self.raw_sequences,
            test_raw_sequences_dict=dataset.test_raw_sequences_dict,
        )
        self.item_meta_le, self.item_meta_dict = process_metadata(
            self.item_metadata,
            exclude_metadata_columns=dataset.exclude_item_metadata_columns,
        )
        self.seq_meta_le, self.seq_meta_dict = process_metadata(
            self.seq_metadata,
            exclude_metadata_columns=dataset.exclude_seq_metadata_columns,
        )

        self.num_seq = len(self.seq_le.classes_)
        self.num_item = len(self.item_le.classes_)
        self.num_item_meta = len(self.item_meta_le.classes_)
        self.num_seq_meta = len(self.seq_meta_le.classes_)

        # seqのmetadataの個数が全て一緒であると仮定している
        self.num_seq_meta_types = (
            len(next(iter(self.seq_metadata.values())))
            if len(self.seq_metadata) > 0
            else 0
        )

        # itemのmetadataの個数が全て一緒であると仮定している
        self.num_item_meta_types = (
            len(next(iter(self.item_metadata.values())))
            if len(self.item_metadata) > 0
            else 0
        )

        print(
            f"num_seq: {self.num_seq}, num_item: {self.num_item}, "
            + f"num_item_meta: {self.num_item_meta}, "
            + f"num_seq_meta: {self.num_seq_meta}, "
            + f"num_item_meta_types: {self.num_item_meta_types}, "
            + f"num_seq_meta_types: {self.num_seq_meta_types}"
        )

        self.train_dataset, self.valid_dataset = create_train_valid_dataset(
            raw_sequences=dataset.train_raw_sequences,
            seq_le=self.seq_le,
            item_le=self.item_le,
            window_size=window_size,
        )

        self.item_meta_indices, self.item_meta_weights = get_meta_indices(
            names=self.item_le.classes_,
            meta_le=self.item_meta_le,
            metadata=self.item_metadata,
            exclude_metadata_columns=dataset.exclude_item_metadata_columns,
        )
        self.seq_meta_indices, self.seq_meta_weights = get_meta_indices(
            names=self.seq_le.classes_,
            meta_le=self.seq_meta_le,
            metadata=self.seq_metadata,
            exclude_metadata_columns=dataset.exclude_seq_metadata_columns,
        )

        self.item_counter: collections.Counter = collections.Counter()
        for sequence in self.train_dataset.sequences:
            for item in sequence:
                self.item_counter[item] += 1

        self.test_datasets = create_test_datasets(
            test_raw_sequences_dict=dataset.test_raw_sequences_dict,
            seq_le=self.seq_le,
            item_le=self.item_le,
            left_window_size=window_size * 2,
        )


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[list[int]],
        data: list[tuple[int, int]],
        left_window_size: int,
        right_window_size: int,
    ) -> None:
        """
        補助情報を含んだシーケンシャルのデータを保持するクラス

        Args:
            raw_sequences (Dict[str, List[str]]):
                生のシーケンシャルデータ
                系列ID : [要素1, 要素2, ..., 要素n]
                例: "doc_001", [ "私", "は", "猫" ]
            seq_le (preprocessing.LabelEncoder):
                系列の名前とindexを対応づけるLabelEncoder
            item_le (preprocessing.LabelEncoder):
                要素の名前とindexを対応づけるLabelEncoder
            window_size (int, optional):
                学習するときに参照する過去の要素の個数.
                Defaults to 8.
        """
        self.sequences = sequences
        self.data = data
        self.left_w = left_window_size
        self.right_w = right_window_size

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[int, list[int], int]:
        seq_index, target_index = self.data[idx]
        item_indices = (
            self.sequences[seq_index][target_index - self.left_w : target_index]  # noqa
            + self.sequences[seq_index][
                target_index + 1 : target_index + self.right_w + 1  # noqa
            ]
        )
        target_index = self.sequences[seq_index][target_index]
        return seq_index, item_indices, target_index


def create_train_valid_dataset(
    raw_sequences: dict[str, list[str]],
    seq_le: preprocessing.LabelEncoder,
    item_le: preprocessing.LabelEncoder,
    window_size: int = 8,
) -> tuple[SequenceDataset, SequenceDataset]:
    sequences, train_data, valid_data = to_sequential_data(
        raw_sequences=raw_sequences,
        seq_le=seq_le,
        item_le=item_le,
        left_window_size=window_size,
        right_window_size=window_size,
    )
    train_dataset = SequenceDataset(
        sequences=sequences,
        data=train_data,
        left_window_size=window_size,
        right_window_size=window_size,
    )
    valid_dataset = SequenceDataset(
        sequences=sequences,
        data=valid_data,
        left_window_size=window_size,
        right_window_size=window_size,
    )
    return train_dataset, valid_dataset


def to_sequential_data(
    raw_sequences: dict[str, list[str]],
    seq_le: preprocessing.LabelEncoder,
    item_le: preprocessing.LabelEncoder,
    left_window_size: int,
    right_window_size: int,
    valid_ratio: float = 0.2,
) -> tuple[list[list[int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """
    シーケンシャルデータを学習データと検証データに変換する
    """
    sequences: list[list[int]] = [[] for _ in range(len(seq_le.classes_))]
    train_data = []
    valid_data = []

    seq_indices = seq_le.transform(list(raw_sequences.keys()))
    for i, raw_sequence in enumerate(raw_sequences.values()):
        seq_index = seq_indices[i]

        sequence = item_le.transform(raw_sequence).tolist()
        sequences[seq_index] = sequence

        # 系列長をLとして、以下を満たすなら検証データを作る
        # L * valid_ratio > left_window_size + right_window_size
        # L * (1 - valid_ratio) > (left_window_size + right_window_size) * 2
        seq_len = len(sequence)
        valid_len = int(round(seq_len * valid_ratio))
        train_len = seq_len - valid_len
        if (
            valid_len > left_window_size + right_window_size
            and train_len > (left_window_size + right_window_size) * 2
        ):
            # 検証データを作る
            train_left = left_window_size
            train_right = train_len - right_window_size
            assert train_left <= train_right
            train_data.extend(
                list(map(lambda j: (seq_index, j), range(train_left, train_right)))
            )
            valid_left = train_len + left_window_size
            valid_right = seq_len - right_window_size
            assert valid_left <= valid_right
            valid_data.extend(
                list(map(lambda j: (seq_index, j), range(valid_left, valid_right)))
            )
        else:
            # 検証データを作らない
            left = left_window_size
            right = seq_len - right_window_size
            if left >= right:
                continue
            train_data.extend(list(map(lambda j: (seq_index, j), range(left, right))))

    return sequences, train_data, valid_data


def create_test_datasets(
    test_raw_sequences_dict: dict[str, dict[str, list[str]]],
    seq_le: preprocessing.LabelEncoder,
    item_le: preprocessing.LabelEncoder,
    left_window_size: int,
) -> dict[str, list[tuple[int, list[int], list[int]]]]:
    return {
        test_name: to_sequential_test_data(
            raw_sequences=raw_sequences,
            seq_le=seq_le,
            item_le=item_le,
            left_window_size=left_window_size,
        )
        for test_name, raw_sequences in test_raw_sequences_dict.items()
    }


def to_sequential_test_data(
    raw_sequences: dict[str, list[str]],
    seq_le: preprocessing.LabelEncoder,
    item_le: preprocessing.LabelEncoder,
    left_window_size: int,
) -> list[tuple[int, list[int], list[int]]]:
    """
    シーケンシャルデータをテストデータに変換する

    Returns:
        (seq_index, context_items, target_items)
    """
    data = []

    seq_indices = seq_le.transform(list(raw_sequences.keys()))
    for i, raw_sequence in enumerate(raw_sequences.values()):
        seq_index = seq_indices[i]
        sequence = item_le.transform(raw_sequence).tolist()

        # 長さが足りていない系列はテストデータに加えない
        if len(sequence) < left_window_size + 1:
            continue

        # テストデータは周囲の要素ではなく、直前の要素のみ特徴量に入れることができる
        context_items = sequence[:left_window_size]
        target_items = sequence[left_window_size:]
        data.append((seq_index, context_items, target_items))

    return data


def process_metadata(
    items: dict[str, dict[str, str]],
    exclude_metadata_columns: Optional[list[str]] = None,
) -> tuple[preprocessing.LabelEncoder, dict[str, list[str]]]:
    """Process meta datas

    Args:
        items (Dict[str, Dict[str, str]]):
            item data (item_id, (meta_name, meta_value))

    Returns:
        Tuple[LabelEncoder, Dict[str, List[str]]]:
            (Label Encoder of meta data, Dictionary of list of meta datas)
    """
    meta_dict: dict[str, MutableSet[str]] = {}
    for _, meta_data in items.items():
        for meta_name, meta_value in meta_data.items():
            if (
                exclude_metadata_columns is not None
                and meta_name in exclude_metadata_columns
            ):
                continue
            if meta_name not in meta_dict:
                meta_dict[meta_name] = set()

            # 補助情報が複数ある場合はリストで渡される
            if isinstance(meta_value, list):
                for e in meta_value:
                    meta_dict[meta_name].add(e)
            else:
                meta_dict[meta_name].add(meta_value)

    all_meta_values: list[str] = []
    new_meta_dict: dict[str, list[str]] = {}
    for meta_name in meta_dict.keys():
        new_meta_dict[meta_name] = list(meta_dict[meta_name])
        for value in new_meta_dict[meta_name]:
            # create str that is identical
            all_meta_values.append(to_full_meta_value(meta_name, value))

    meta_le = preprocessing.LabelEncoder().fit(all_meta_values)

    return meta_le, new_meta_dict


def get_meta_indices(
    names: list[str],
    meta_le: preprocessing.LabelEncoder,
    metadata: dict[str, dict[str, Any]],
    exclude_metadata_columns: Optional[list[str]] = None,
    max_meta_size: int = 10,
) -> tuple[Tensor, Tensor]:
    meta_indices: list[list[int]] = []
    meta_weights: list[list[float]] = []
    for name in names:
        if name not in metadata:
            meta_indices.append([])
            meta_weights.append([])
            continue
        meta_values: list[str] = []
        meta_weight: list[float] = []
        for meta_name, meta_value in metadata[name].items():
            if (
                exclude_metadata_columns is not None
                and meta_name in exclude_metadata_columns
            ):
                continue
            if isinstance(meta_value, list):
                for e in meta_value:
                    meta_values.append(to_full_meta_value(meta_name, str(e)))
                    meta_weight.append(1 / len(meta_value))
            else:
                meta_values.append(to_full_meta_value(meta_name, str(meta_value)))
                meta_weight.append(1)

        meta_index = list(meta_le.transform(meta_values))

        assert len(meta_weight) <= max_meta_size
        assert len(meta_weight) == len(meta_index)

        # 大きさをmax_meta_sizeに合わせるために、
        # weightが0であるmeta_indicesを末尾に加える
        while len(meta_weight) < max_meta_size:
            meta_index.append(0)
            meta_weight.append(0)

        meta_indices.append(meta_index)
        meta_weights.append(meta_weight)

    meta_indices = torch.tensor(meta_indices, dtype=torch.long, requires_grad=False)
    meta_weights = torch.tensor(meta_weights, dtype=torch.float, requires_grad=False)
    return meta_indices, meta_weights


def get_seq_item_le(
    raw_sequences: dict[str, list[str]],
    test_raw_sequences_dict: dict[str, dict[str, list[str]]],
) -> tuple[preprocessing.LabelEncoder, preprocessing.LabelEncoder]:
    seq_le = preprocessing.LabelEncoder().fit(
        list(raw_sequences.keys())
        + sum(
            list(
                map(
                    lambda d: list(d.keys()),
                    test_raw_sequences_dict.values(),
                )
            ),
            [],
        )
    )
    item_le = preprocessing.LabelEncoder().fit(
        sum(raw_sequences.values(), [])
        + sum(
            sum(
                list(
                    map(
                        lambda d: list(d.values()),
                        test_raw_sequences_dict.values(),
                    )
                ),
                [],
            ),
            [],
        )
    )
    return seq_le, item_le


def load_dataset_center(
    dataset_name: str,
    window_size: int = 5,
    data_dir: str = "data/",
) -> SequenceDatasetCenter:
    dataset = load_raw_dataset(dataset_name=dataset_name, data_dir=data_dir)
    dataset_center = SequenceDatasetCenter(dataset, window_size=window_size)
    return dataset_center
