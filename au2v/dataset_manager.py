import copy
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, MutableSet, Optional, Tuple

import torch
import tqdm
from sklearn import preprocessing
from torch import Tensor
from torch.utils.data import Dataset

from au2v.dataset import RawDataset, load_raw_dataset
from au2v.util import get_all_items, to_full_meta_value


class SequenceDatasetManager:
    """
    訓練データとテストデータを管理するクラス
    """

    def __init__(self, dataset: RawDataset, window_size: int) -> None:
        self.raw_sequences = copy.deepcopy(dataset.train_raw_sequences)
        if dataset.test_raw_sequences_dict is not None:
            # seq_idが一緒の訓練データとテストデータがあるかもしれないので、系列をマージする
            for _, test_raw_sequences in dataset.test_raw_sequences_dict.items():
                for seq_id, raw_sequence in test_raw_sequences.items():
                    if seq_id in self.raw_sequences:
                        self.raw_sequences[seq_id].extend(raw_sequence)
                    else:
                        self.raw_sequences[seq_id] = raw_sequence

        self.item_metadata = (
            dataset.item_metadata if dataset.item_metadata is not None else {}
        )
        self.seq_metadata = (
            dataset.seq_metadata if dataset.seq_metadata is not None else {}
        )

        items = get_all_items(self.raw_sequences)
        self.item_le = preprocessing.LabelEncoder().fit(items)
        self.item_meta_le, self.item_meta_dict = process_metadata(
            self.item_metadata,
            exclude_metadata_columns=dataset.exclude_item_metadata_columns,
        )
        self.seq_le = preprocessing.LabelEncoder().fit(list(self.raw_sequences.keys()))
        self.seq_meta_le, self.seq_meta_dict = process_metadata(
            self.seq_metadata,
            exclude_metadata_columns=dataset.exclude_seq_metadata_columns,
        )

        self.num_seq = len(self.raw_sequences)
        self.num_item = len(items)
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

        self.train_dataset = SequenceDataset(
            raw_sequences=dataset.train_raw_sequences,
            seq_le=self.seq_le,
            item_le=self.item_le,
            window_size=window_size,
        )
        self.sequences = copy.deepcopy(self.train_dataset.sequences)

        if dataset.test_raw_sequences_dict is not None:
            self.test_datasets: Optional[Dict[str, SequenceDataset]] = {}
            for (
                test_name,
                test_raw_sequences,
            ) in dataset.test_raw_sequences_dict.items():
                self.test_datasets[test_name] = SequenceDataset(
                    raw_sequences=test_raw_sequences,
                    seq_le=self.seq_le,
                    item_le=self.item_le,
                    window_size=window_size,
                )
                self.sequences += self.test_datasets[test_name].sequences
        else:
            self.test_datasets = None

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


class SequenceDataset(Dataset):
    def __init__(
        self,
        raw_sequences: Dict[str, List[str]],
        seq_le: preprocessing.LabelEncoder,
        item_le: preprocessing.LabelEncoder,
        window_size: int = 8,
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
        self.raw_sequences = raw_sequences

        self.sequences, self.data = to_sequential_data(
            raw_sequences=self.raw_sequences,
            seq_le=seq_le,
            item_le=item_le,
            window_size=window_size,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        """
        Returns:
        (
          seq_index,
          target_index
        )
        """
        return self.data[idx]


def process_metadata(
    items: Dict[str, Dict[str, str]],
    exclude_metadata_columns: Optional[List[str]] = None,
) -> Tuple[preprocessing.LabelEncoder, Dict[str, List[str]]]:
    """Process meta datas

    Args:
        items (Dict[str, Dict[str, str]]):
            item data (item_id, (meta_name, meta_value))

    Returns:
        Tuple[LabelEncoder, Dict[str, List[str]]]:
            (Label Encoder of meta data, Dictionary of list of meta datas)
    """
    meta_dict: Dict[str, MutableSet[str]] = {}
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

    all_meta_values: List[str] = []
    new_meta_dict: Dict[str, List[str]] = {}
    for meta_name in meta_dict.keys():
        new_meta_dict[meta_name] = list(meta_dict[meta_name])
        for value in new_meta_dict[meta_name]:
            # create str that is identical
            all_meta_values.append(to_full_meta_value(meta_name, value))

    meta_le = preprocessing.LabelEncoder().fit(all_meta_values)

    return meta_le, new_meta_dict


def get_meta_indices(
    names: List[str],
    meta_le: preprocessing.LabelEncoder,
    metadata: Dict[str, Dict[str, Any]],
    exclude_metadata_columns: Optional[List[str]] = None,
    max_meta_size: int = 10,
) -> Tuple[Tensor, Tensor]:
    meta_indices: List[List[int]] = []
    meta_weights: List[List[float]] = []
    for name in names:
        if name not in metadata:
            meta_indices.append([])
            meta_weights.append([])
            continue
        meta_values: List[str] = []
        meta_weight: List[float] = []
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


def to_sequential_data(
    raw_sequences: Dict[str, List[str]],
    seq_le: preprocessing.LabelEncoder,
    item_le: preprocessing.LabelEncoder,
    window_size: int,
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    シーケンシャルデータを学習データに変換する

    Returns:
        Tuple[List[List[int]], List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]]:
            (sequences, data)
    """
    sequences: List[List[int]] = [[] for _ in range(len(seq_le.classes_))]
    data = []

    print("to_sequential_data start")
    seq_indices = seq_le.transform(list(raw_sequences.keys()))
    for i, raw_sequence in enumerate(tqdm.tqdm(raw_sequences.values())):
        seq_index = seq_indices[i]

        sequence = item_le.transform(raw_sequence).tolist()
        sequences[seq_index] = sequence

        left = window_size
        right = len(sequence) - 1 - window_size
        if left >= right:
            continue
        data.extend(list(map(lambda j: (seq_index, j), range(left, right))))

    print("to_sequential_data end")
    return sequences, data


def load_dataset_manager(
    dataset_name: str,
    dataset_dir: str,
    load_dataset: bool,
    save_dataset: bool,
    window_size: int = 5,
    data_dir: str = "data/",
) -> SequenceDatasetManager:
    pickle_path = Path(dataset_dir).joinpath(f"{dataset_name}.pickle")

    if load_dataset and os.path.exists(pickle_path):
        print(f"load cached dataset_manager from: {pickle_path}")
        with open(pickle_path, "rb") as f:
            dataset_manager: SequenceDatasetManager = pickle.load(f)
        return dataset_manager

    print(f"dataset_manager does not exist at: {pickle_path}, create dataset")

    dataset = load_raw_dataset(dataset_name=dataset_name, data_dir=data_dir)

    dataset_manager = SequenceDatasetManager(dataset, window_size=window_size)

    if save_dataset:
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"dumping dataset_manager to: {pickle_path}")
        with open(pickle_path, "wb") as f:
            pickle.dump(dataset_manager, f)
        print(f"dumped dataset_manager to: {pickle_path}")

    return dataset_manager
