from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainerConfig:
    model_name: str
    epochs: int
    batch_size: int
    load_model: bool
    verbose: bool
    model_path: Optional[str] = None


@dataclass
class ModelConfig:
    d_model: int
    window_size: int
    negative_sample_size: int
    lr: float
    use_learnable_embedding: bool


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--num_cluster", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    return parser.parse_args()


def setup_config(args: Namespace) -> Tuple[TrainerConfig, ModelConfig]:
    trainer_config = TrainerConfig(
        model_name="attentive",
        epochs=args.epochs,
        batch_size=args.batch_size,
        load_model=args.load_model,
        verbose=args.verbose,
        model_path=None,
    )
    model_config = ModelConfig(
        d_model=args.d_model,
        window_size=8,
        negative_sample_size=5,
        lr=args.lr,
        use_learnable_embedding=False,
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
    )
    return trainer_config, model_config
