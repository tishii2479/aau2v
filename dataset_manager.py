import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import tqdm
from sklearn import preprocessing
from torch import Tensor
from torch.utils.data import Dataset

from util import get_all_items, to_full_meta_value


class SequenceDatasetManager:
    def __init__(
        self,
        train_raw_sequences: Dict[str, List[str]],
        item_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        seq_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
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

        if len(self.seq_metadata) > 0:
            # seqのmetadataの個数が全て一緒であると仮定している
            self.num_seq_meta_types = len(next(iter(self.seq_metadata.values())))
        else:
            self.num_seq_meta_types = 0

        if len(self.item_metadata) > 0:
            # itemのmetadataの個数が全て一緒であると仮定している
            self.num_item_meta_types = len(next(iter(self.item_metadata.values())))
        else:
            self.num_item_meta_types = 0

        print(
            f"num_seq: {self.num_seq}, num_item: {self.num_item}, "
            + f"num_item_meta: {self.num_item_meta}, "
            + f"num_seq_meta: {self.num_seq_meta}, "
            + f"num_item_meta_types: {self.num_item_meta_types}"
        )

        self.train_dataset = SequenceDataset(
            raw_sequences=train_raw_sequences,
            seq_le=self.seq_le,
            item_le=self.item_le,
            window_size=window_size,
        )
        self.sequences = self.train_dataset.sequences

        if test_raw_sequences_dict is not None:
            self.test_dataset: Optional[Dict[str, SequenceDataset]] = {}
            for test_name, test_raw_sequences in test_raw_sequences_dict.items():
                self.test_dataset[test_name] = SequenceDataset(
                    raw_sequences=test_raw_sequences,
                    seq_le=self.seq_le,
                    item_le=self.item_le,
                    window_size=window_size,
                )
                self.sequences += self.test_dataset[test_name].sequences
        else:
            self.test_dataset = None

        self.item_meta_indicies, self.item_meta_weights = get_meta_indicies(
            names=self.item_le.classes_,
            meta_le=self.item_meta_le,
            metadata=self.item_metadata,
            exclude_metadata_columns=exclude_item_metadata_columns,
        )
        self.seq_meta_indicies, self.seq_meta_weights = get_meta_indicies(
            names=self.seq_le.classes_,
            meta_le=self.seq_meta_le,
            metadata=self.seq_metadata,
            exclude_metadata_columns=exclude_seq_metadata_columns,
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
            item_metadata (Dict[str, Dict[str, Any]]):
                要素の補助情報の辞書
                要素 : {
                    補助情報ID: 補助情報の値
                }
                例: "私" : {
                    "品詞": "名詞",
                    "長さ": 1
                }
            seq_metadata (Dict[str, Dict[str, Any]]):
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
            seq_le=seq_le,
            item_le=item_le,
            window_size=window_size,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
        (
          seq_index,
          item_indicies,
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

            # 補助情報が複数ある場合はリストで渡される
            if isinstance(meta_value, list):
                for e in meta_value:
                    meta_dict[meta_name].add(e)
            else:
                meta_dict[meta_name].add(meta_value)

    all_meta_values: List[str] = []
    for meta_name, meta_values in meta_dict.items():
        for value in meta_values:
            # create str that is identical
            all_meta_values.append(to_full_meta_value(meta_name, value))

    meta_le = preprocessing.LabelEncoder().fit(all_meta_values)

    return meta_le, meta_dict


def get_meta_indicies(
    names: List[str],
    meta_le: preprocessing.LabelEncoder,
    metadata: Dict[str, Dict[str, Any]],
    exclude_metadata_columns: Optional[List[str]] = None,
    max_meta_size: int = 10,
) -> Tuple[Tensor, Tensor]:
    meta_indicies: List[List[int]] = []
    meta_weights: List[List[float]] = []
    for name in names:
        if name not in metadata:
            meta_indicies.append([])
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
        # weightが0であるmeta_indiciesを末尾に加える
        while len(meta_weight) < max_meta_size:
            meta_index.append(0)
            meta_weight.append(0)

        meta_indicies.append(meta_index)
        meta_weights.append(meta_weight)

    meta_indicies = torch.tensor(meta_indicies, dtype=torch.long, requires_grad=False)
    meta_weights = torch.tensor(meta_weights, dtype=torch.float, requires_grad=False)
    return meta_indicies, meta_weights


def to_sequential_data(
    raw_sequences: Dict[str, List[str]],
    seq_le: preprocessing.LabelEncoder,
    item_le: preprocessing.LabelEncoder,
    window_size: int,
) -> Tuple[List[List[int]], List[Tuple[Tensor, Tensor, Tensor]]]:
    """
    シーケンシャルデータを学習データに変換する

    Returns:
        Tuple[List[List[int]], List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]]:
            (sequences, data)
    """
    sequences: List[List[int]] = []
    data = []

    # TODO: parallelize
    print("to_sequential_data start")
    for seq_name, raw_sequence in tqdm.tqdm(raw_sequences.items()):
        sequence = item_le.transform(raw_sequence)
        sequences.append(sequence)

        seq_index = torch.tensor(seq_le.transform([seq_name])[0], dtype=torch.long)
        for j in range(len(sequence) - window_size):
            item_indicies = torch.tensor(
                sequence[j : j + window_size], dtype=torch.long
            )
            target_index = torch.tensor(sequence[j + window_size], dtype=torch.long)

            data.append(
                (
                    seq_index,
                    item_indicies,
                    target_index,
                )
            )
    print("to_sequential_data end")
    return sequences, data
