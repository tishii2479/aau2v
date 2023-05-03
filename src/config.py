from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class TrainerConfig:
    model_name: str = "attentive"
    dataset_name: str = "toydata-small"
    epochs: int = 3
    batch_size: int = 64
    verbose: bool = False
    ignore_saved_model: bool = False
    load_model: bool = True
    save_model: bool = True
    load_dataset: bool = True
    save_dataset: bool = True
    model_dir: str = "cache/model/"
    dataset_dir: str = "cache/dataset/"

    @property
    def model_path(self) -> str:
        # ISSUE: stop creating directory if model_dir does not exist
        model_dir = Path(self.model_dir, self.dataset_name)
        model_dir.mkdir(exist_ok=True, parents=True)
        return str(model_dir.joinpath(f"{self.model_name}.pt"))

    @property
    def best_model_path(self) -> str:
        # ISSUE: stop creating directory if best_model_path does not exist
        model_dir = Path(self.model_dir, self.dataset_name)
        model_dir.mkdir(exist_ok=True, parents=True)
        return str(model_dir.joinpath(f"best-{self.model_name}.pt"))


@dataclass
class ModelConfig:
    d_model: int = 128
    init_embedding_std: float = 0.2
    max_embedding_norm: Optional[float] = None
    window_size: int = 5
    negative_sample_size: int = 5
    lr: float = 0.0001


def parse_config() -> Tuple[TrainerConfig, ModelConfig]:
    parser = ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="attentive",
        help="使用するモデル",
        choices=["attentive", "old-attentive", "doc2vec"],
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="toydata-paper",
        help="使用するデータセット",
        choices=[
            "toydata-paper",
            "toydata-small",
            "hm",
            "movielens",
            "movielens-simple",
            "movielens-equal-gender",
            "20newsgroup",
            "20newsgroup-small",
        ],
    )
    parser.add_argument("--num-cluster", type=int, default=10, help="クラスタリングの際に使うクラスタ数")
    parser.add_argument("--d-model", type=int, default=128, help="埋め込み表現の次元数")
    parser.add_argument(
        "--max-embedding-norm",
        type=float,
        default=None,
        help="埋め込み表現のノルムの最大値",
    )
    parser.add_argument(
        "--init-embedding-std",
        type=float,
        default=0.2,
        help="埋め込み表現を初期化する時に用いる正規分布の標準偏差",
    )
    parser.add_argument("--window-size", type=int, default=5, help="学習する際に参照する過去の要素の個数")
    parser.add_argument(
        "--negative_sample_size", type=int, default=5, help="ネガティブサンプリングのサンプル数"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument("--lr", type=float, default=0.0001, help="学習率")
    parser.add_argument("--verbose", action="store_true", help="ログを詳細に出すかどうか")
    parser.add_argument(
        "--load-model", action="store_true", help="`model_dir`からモデルのパラメータを読み込むかどうか"
    )
    parser.add_argument(
        "--ignore-saved-model",
        action="store_true",
        help="`model_dir`にあるモデルのパラメータを無視するかどうか",
    )
    parser.add_argument(
        "--no-save-model", action="store_true", help="`model_dir`にモデルを保存するかどうか"
    )
    parser.add_argument(
        "--no-load-dataset", action="store_true", help="`datset_dir`からデータセットを読み込むかどうか"
    )
    parser.add_argument(
        "--no-save-dataset", action="store_true", help="`dataset_dir`にデータセットを保存するかどうか"
    )
    parser.add_argument(
        "--model-dir", type=str, default="cache/model/", help="モデルを保存するディレクトリ"
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
        ignore_saved_model=args.ignore_saved_model,
        save_model=(args.no_save_model is False),
        load_dataset=(args.no_load_dataset is False),
        save_dataset=(args.no_save_dataset is False),
        verbose=args.verbose,
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
    )
    model_config = ModelConfig(
        d_model=args.d_model,
        init_embedding_std=args.init_embedding_std,
        max_embedding_norm=args.max_embedding_norm,
        window_size=args.window_size,
        negative_sample_size=args.negative_sample_size,
        lr=args.lr,
    )
    return trainer_config, model_config
