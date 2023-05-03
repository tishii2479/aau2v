from argparse import ArgumentParser
from typing import Tuple

import numpy as np
import torch

from src.config import ModelConfig, TrainerConfig
from src.dataset import load_dataset_manager
from src.trainers import PyTorchTrainer


def main() -> None:
    torch.manual_seed(24)
    np.random.seed(24)

    trainer_config, model_config = parse_config()
    print("trainer_config:", trainer_config)
    print("model_config:", model_config)

    dataset_manager = load_dataset_manager(
        dataset_name=trainer_config.dataset_name,
        dataset_dir=trainer_config.dataset_dir,
        load_dataset=trainer_config.load_dataset,
        save_dataset=trainer_config.save_dataset,
        window_size=model_config.window_size,
    )

    trainer = PyTorchTrainer(
        dataset_manager=dataset_manager,
        trainer_config=trainer_config,
        model_config=model_config,
    )
    trainer.fit(show_fig=False)


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


if __name__ == "__main__":
    main()
