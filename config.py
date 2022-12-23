from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainerConfig:
    model_name: str = "attentive"
    dataset_name: str = "toydata-small"
    epochs: int = 5
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
        # create directory if model_dir does not exist
        model_dir = Path(self.model_dir, self.dataset_name)
        model_dir.mkdir(exist_ok=True, parents=True)
        return str(model_dir.joinpath(f"{self.model_name}.pt"))

    @property
    def best_model_path(self) -> str:
        # create directory if model_dir does not exist
        model_dir = Path(self.model_dir, self.dataset_name)
        model_dir.mkdir(exist_ok=True, parents=True)
        return str(model_dir.joinpath(f"best-{self.model_name}.pt"))


@dataclass
class ModelConfig:
    d_model: int = 32
    init_embedding_std: float = 1
    max_embedding_norm: Optional[float] = None
    window_size: int = 8
    negative_sample_size: int = 5
    lr: float = 0.001
    normalize_embedding_weight: bool = False
