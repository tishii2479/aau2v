from dataclasses import dataclass
from typing import Optional


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
