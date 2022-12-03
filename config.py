from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class TrainerConfig:
    model_name: str = "attentive2"
    dataset_name: str = "20newsgroup-small"
    epochs: int = 5
    batch_size: int = 16
    verbose: bool = False
    load_model: bool = True
    save_model: bool = True
    load_dataset: bool = True
    save_dataset: bool = True
    cache_dir: str = "cache/"
    dataset_dir: str = "cache/dataset/"

    @property
    def model_path(self) -> str:
        return str(Path(self.cache_dir, f"{self.dataset_name}-{self.model_name}.pt"))

    @property
    def best_model_path(self) -> str:
        return str(
            Path(self.cache_dir, f"best-{self.dataset_name}-{self.model_name}.pt")
        )


@dataclass
class ModelConfig:
    d_model: int = 64
    max_embedding_norm: Optional[float] = None
    window_size: int = 8
    negative_sample_size: int = 5
    lr: float = 0.01
    use_learnable_embedding: bool = True
    dropout: float = 0.1
    add_seq_embedding: bool = False
    add_positional_encoding: bool = False


def parse_config() -> Tuple[TrainerConfig, ModelConfig]:
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, default="attentive", help="使用するモデル")
    parser.add_argument(
        "--dataset-name", type=str, default="attentive", help="使用するデータセット"
    )
    parser.add_argument("--num-cluster", type=int, default=10, help="クラスタリングの際に使うクラスタ数")
    parser.add_argument("--d-model", type=int, default=50, help="埋め込み表現の次元数")
    parser.add_argument(
        "--max-embedding-norm",
        type=float,
        default=None,
        help="埋め込み表現のノルムの最大値",
    )
    parser.add_argument("--window-size", type=int, default=8, help="学習する際に参照する過去の要素の個数")
    parser.add_argument(
        "--negative_sample_size", type=int, default=5, help="ネガティブサンプリングのサンプル数"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=5, help="エポック数")
    parser.add_argument("--lr", type=float, default=0.01, help="学習率")
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="位置エンコーディング時にドロップアウトする割合"
    )
    parser.add_argument(
        "--add-seq-embedding", action="store_true", help="予測ベクトルに系列の埋め込み表現を足すかどうか"
    )
    parser.add_argument(
        "--add-positional-encoding",
        action="store_true",
        help="位置エンコーディングを要素の埋め込み表現に足すかどうか",
    )
    parser.add_argument(
        "--use-learnable-embedding",
        action="store_true",
        help="要素の埋め込み表現を学習可能にするかどうか（experimental）",
    )
    parser.add_argument("--verbose", action="store_true", help="ログを詳細に出すかどうか")
    parser.add_argument(
        "--load-model", action="store_true", help="`cache_dir`からモデルのパラメータを読み込むかどうか"
    )
    parser.add_argument(
        "--no-save-model", action="store_true", help="`cache_dir`にモデルを保存するかどうか"
    )
    parser.add_argument(
        "--no-load-dataset", action="store_true", help="`cache_dir`からデータセットを読み込むかどうか"
    )
    parser.add_argument(
        "--no-save-dataset", action="store_true", help="`dataset_dir`にデータセットを保存するかどうか"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="cache/", help="モデルを保存するディレクトリ"
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="cache/dataset/", help="データセットを保存するディレクトリ"
    )

    args = parser.parse_args()

    trainer_config = TrainerConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        load_model=args.load_model,
        save_model=(args.no_save_model is False),
        load_dataset=(args.no_load_dataset is False),
        save_dataset=(args.no_save_dataset is False),
        verbose=args.verbose,
        cache_dir=args.cache_dir,
        dataset_dir=args.dataset_dir,
    )
    model_config = ModelConfig(
        d_model=args.d_model,
        max_embedding_norm=args.max_embedding_norm,
        window_size=args.window_size,
        negative_sample_size=args.negative_sample_size,
        lr=args.lr,
        use_learnable_embedding=args.use_learnable_embedding,
        dropout=args.dropout,
        add_seq_embedding=args.add_seq_embedding,
        add_positional_encoding=args.add_positional_encoding,
    )
    return trainer_config, model_config
