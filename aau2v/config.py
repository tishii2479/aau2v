from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainerConfig:
    model_name: str = "aau2v"
    dataset_name: str = "toydata-paper"
    epochs: int = 3
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0
    verbose: bool = False
    model_dir: str = "model"
    save_model: bool = True
    device: str = "cpu"


@dataclass
class ModelConfig:
    d_model: int = 64
    init_embedding_std: float = 0.2
    max_embedding_norm: Optional[float] = 5
    window_size: int = 5
    negative_sample_size: int = 5
    use_weight_tying: bool = True
    use_meta: bool = True
    use_attention: bool = True


def parse_config() -> tuple[TrainerConfig, ModelConfig]:
    parser = ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="aau2v",
        help="The name of embedding model to train",
        choices=["aau2v", "user2vec"],
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="toydata-paper",
        help="The name of dataset to use for training",
        choices=[
            "toydata-paper",
            "toydata-small",
            "movielens",
        ],
    )
    parser.add_argument(
        "--d-model", type=int, default=64, help="The dimension of embeddings"
    )
    parser.add_argument(
        "--max-embedding-norm",
        type=float,
        default=5,
        help="The maximum l2-norm of embedding representations",
    )
    parser.add_argument(
        "--init-embedding-std",
        type=float,
        default=0.2,
        help="The standard deviation of the normal distribution used when initializing embeddings",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="The number of elements referenced during training (window_size)",
    )
    parser.add_argument(
        "--negative_sample_size",
        type=int,
        default=5,
        help="The sample size for negative sampling",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="The learning rate when optimizing"
    )
    parser.add_argument(
        "--no-weight-tying",
        action="store_true",
        help="Train model without weight-tying",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Train model without auxiliary information",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Train model without attention aggregation",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="The batch size")
    parser.add_argument(
        "--epochs", type=int, default=3, help="The epoch number for training"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="The strenght of weight decay.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to output logs in detail or not"
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Does not save model",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="model",
        help="The directory to save model weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device on which the computation is performed",
    )

    args = parser.parse_args()

    trainer_config = TrainerConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        verbose=args.verbose,
        model_dir=args.model_dir,
        save_model=not args.no_save_model,
        device=args.device,
    )
    model_config = ModelConfig(
        d_model=args.d_model,
        init_embedding_std=args.init_embedding_std,
        max_embedding_norm=args.max_embedding_norm,
        window_size=args.window_size,
        negative_sample_size=args.negative_sample_size,
        use_weight_tying=args.use_weight_tying,
        use_meta=args.use_meta,
        use_attention=args.use_attention,
    )
    return trainer_config, model_config
