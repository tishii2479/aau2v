from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainerConfig:
    model_name: str = "attentive2"
    dataset_name: str = "20newsgroup-small"
    epochs: int = 5
    batch_size: int = 64
    verbose: bool = False
    ignore_saved_model: bool = False
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
    lr: float = 0.005
    use_learnable_embedding: bool = True
    dropout: float = 0.1
    add_seq_embedding: bool = False
    add_positional_encoding: bool = False
