from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainerConfig:
    model_name: str
    epochs: int
    batch_size: int
    verbose: bool
    load_model: bool
    model_path: Optional[str] = None


@dataclass
class ModelConfig:
    d_model: int
    window_size: int
    negative_sample_size: int
    lr: float
    use_learnable_embedding: bool
    dropout: float = 0.1
    add_seq_embedding: bool = True
    add_positional_encoding: bool = False


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="attentive", help="使用するモデル")
    parser.add_argument("--num_cluster", type=int, default=10, help="クラスタリングの際に使うクラスタ数")
    parser.add_argument("--d_model", type=int, default=50, help="埋め込み表現の次元数")
    parser.add_argument("--window_size", type=int, default=8, help="学習する際に参照する過去の要素の個数")
    parser.add_argument(
        "--negative_sample_size", type=int, default=5, help="ネガティブサンプリングのサンプル数"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=5, help="エポック数")
    parser.add_argument("--lr", type=float, default=0.0005, help="学習率")
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="位置エンコーディング時にドロップアウトする割合"
    )
    parser.add_argument(
        "--add_seq_embedding", action="store_true", help="予測ベクトルに系列の埋め込み表現を足すかどうか"
    )
    parser.add_argument(
        "--add_positional_encoding",
        action="store_true",
        help="位置エンコーディングを要素の埋め込み表現に足すかどうか",
    )
    parser.add_argument(
        "--use_learnable_embedding",
        action="store_true",
        help="要素の埋め込み表現を学習可能にするかどうか（experimental）",
    )
    parser.add_argument("--verbose", action="store_true", help="ログを詳細に出すかどうか")
    parser.add_argument(
        "--load_model", action="store_true", help="`model_path`からモデルのパラメータを読み込むかどうか"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="読み込むモデルのパラメータのファイルパス"
    )
    return parser.parse_args()


def setup_config(args: Namespace) -> Tuple[TrainerConfig, ModelConfig]:
    trainer_config = TrainerConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        load_model=args.load_model,
        verbose=args.verbose,
        model_path=args.model_path,
    )
    model_config = ModelConfig(
        d_model=args.d_model,
        window_size=args.window_size,
        negative_sample_size=args.negative_sample_size,
        lr=args.lr,
        use_learnable_embedding=args.use_learnable_embedding,
        dropout=args.dropout,
        add_seq_embedding=args.add_seq_embedding,
        add_positional_encoding=args.add_positional_encoding,
    )
    return trainer_config, model_config


def default_config() -> Tuple[TrainerConfig, ModelConfig]:
    trainer_config = TrainerConfig(
        model_name="attentive",
        epochs=2,
        batch_size=10,
        load_model=False,
        verbose=False,
        model_path=None,
    )
    model_config = ModelConfig(
        d_model=50,
        window_size=8,
        negative_sample_size=5,
        lr=0.0005,
        use_learnable_embedding=False,
        add_seq_embedding=False,
        add_positional_encoding=False,
    )
    return trainer_config, model_config
